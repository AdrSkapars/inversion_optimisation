from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import json
import random
import numpy as np
from tqdm import tqdm
import logging

from inversion_optimisation.utils import get_paper_summary_stats_new


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


class LLMInversionModel(nn.Module):
    """
    This model takes logits from a causal LM and predicts the original input
    that would have generated those logits.
    """
    def __init__(
        self,
        t5_model_name="t5-base",
        t5_tokenizer_name="t5-base",
        llm_model_name="meta-llama/Llama-2-7b-hf",
        unigram_beta=0.01,
        num_tokens=64,
        bottleneck_dim=4096,
    ):
        super().__init__()

        # Load base models
        self.encoder_decoder = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)

        # Load tokenizers models
        self.tokenizer = AutoTokenizer.from_pretrained(t5_tokenizer_name)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # Freeze the LLM
        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm.eval()

        # Get dimensions
        self.encoder_hidden_dim = self.encoder_decoder.config.d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_tokens = num_tokens

        self.mini_batch_size = None

        # Calculate padding to ensure dimensions align
        self.num_zeros_to_add = (num_tokens - (self.llm.config.vocab_size % num_tokens)) % num_tokens

        # Prepare unigram adaptation mechanism
        self.unigram_beta = unigram_beta
        self.unigram = nn.Parameter(
            torch.zeros(
                (1, self.llm.config.vocab_size + self.num_zeros_to_add),
                dtype=torch.float32
            ),
            requires_grad=False
        )

        # Prepare word embeddings with proper reshaping
        word_embeddings = self.encoder_decoder.encoder.embed_tokens.weight.detach().clone()
        self.word_embeddings = self._prepare_word_embeddings(word_embeddings, num_tokens)

        # Prepare transformation from reduced logits to encoder space
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, bottleneck_dim),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim)
        )


    def _prepare_word_embeddings(self, word_embeddings, num_tokens):
        """ Prepare word embeddings with proper padding and reshaping """
        num_zeros_to_add = (num_tokens - (word_embeddings.shape[0] % num_tokens)) % num_tokens

        word_embedding_zeros = torch.zeros(
            (num_zeros_to_add, word_embeddings.shape[1]),
            dtype=torch.float32,
            device=word_embeddings.device
        )

        padded_word_embeddings = torch.cat((word_embeddings, word_embedding_zeros), dim=0)
        reshaped_embeddings = padded_word_embeddings.reshape(
            (num_tokens, -1, word_embeddings.shape[1])
        )

        return nn.Parameter(reshaped_embeddings, requires_grad=False)


    def get_llm_logits(self, input_ids, attention_mask):
        """ Get logits from the LLM model """
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)

            # Get the logits for the last token of each sequence
            batch_size = input_ids.shape[0]
            last_token_indices = attention_mask.sum(dim=1) - 1
            logits = outputs.logits[torch.arange(batch_size), last_token_indices]

        return logits


    def process_logits(self, logits, batch_size):
        """ Process logits through the unigram system and add padding """
        # Apply the unigram adaptation (and update it if in training)
        if self.training:
            unigram_batch = logits.mean(dim=0, keepdim=True)
            if self.unigram.sum() == 0:
                self.unigram.data = unigram_batch
            else:
                self.unigram.data = self.unigram.data * (1 - self.unigram_beta) + unigram_batch * self.unigram_beta
        logits = logits - self.unigram[:, :logits.shape[1]]

        # Add zeros padding
        zeros = torch.zeros(
            (batch_size, self.num_zeros_to_add),
            dtype=logits.dtype,
            device=logits.device
        )
        return torch.cat((logits, zeros), dim=1)


    def map_logits_to_encoder_space(self, embeddings):
        """ Map the processed logits to the encoder embedding space """
        batch_size = embeddings.shape[0]

        # Calculate how many embeddings per token (vocabulary / num_tokens)
        embs_per_token = embeddings.shape[1] // self.num_tokens

        # Reshape logits to match expected format: [batch, num_tokens, embs_per_token]
        embeddings = embeddings.reshape(batch_size, self.num_tokens, embs_per_token)

        # Process in minibatches for memory efficiency
        if self.mini_batch_size is None:
            self.mini_batch_size = min(128, batch_size)
        embeddings_list = []
        for i in range(0, batch_size, self.mini_batch_size):
            end_idx = min(i + self.mini_batch_size, batch_size)
            batch_logits = embeddings[i:end_idx]

            # For each token position, we need to transform the logits
            token_embeddings = []

            for token_idx in range(self.num_tokens):
                # Get the word embeddings for this token position
                token_word_embs = self.word_embeddings[token_idx]  # [vocab_per_token, emb_dim]

                # Get the logits for this token position
                token_logits = batch_logits[:, token_idx]  # [batch, embs_per_token]

                # We need to ensure dimensions match for the matrix multiplication
                if token_word_embs.shape[0] != embs_per_token:
                    # Pad or truncate to match
                    if token_word_embs.shape[0] > embs_per_token:
                        token_word_embs = token_word_embs[:embs_per_token]
                    else:
                        pad_size = embs_per_token - token_word_embs.shape[0]
                        padding = torch.zeros(pad_size, token_word_embs.shape[1],
                                             device=token_word_embs.device)
                        token_word_embs = torch.cat([token_word_embs, padding], dim=0)

                # Matrix multiply: [batch, embs_per_token] @ [embs_per_token, emb_dim] -> [batch, emb_dim]
                token_embedding = torch.matmul(token_logits, token_word_embs)
                token_embeddings.append(token_embedding)

            # Stack along sequence dimension
            batch_embeddings = torch.stack(token_embeddings, dim=1)  # [batch, num_tokens, emb_dim]
            embeddings_list.append(batch_embeddings)

        return torch.cat(embeddings_list, dim=0)


    def _shift_right(self, input_ids):
        """
        Shift input ids one token to the right, and prepend with the pad token.
        This is a helper method for preparing decoder inputs for T5.
        """
        # Create a tensor full of pad tokens to prepend
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.tokenizer.pad_token_id

        return shifted_input_ids


    def forward(self, input_ids, attention_mask=None, labels=None):
        """ Forward pass of the inversion model """
        batch_size = input_ids.shape[0]

        # Create attention mask if none provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Step 1: Get logits from LLM
        logits = self.get_llm_logits(input_ids, attention_mask)

        # Step 2: Process logits (unigram adaptation and padding)
        processed_logits = self.process_logits(logits, batch_size)

        # Step 3: Map to encoder space
        encoder_embeddings = self.map_logits_to_encoder_space(processed_logits)

        # Step 4: Create encoder attention mask
        encoder_attention_mask = torch.ones(
            (batch_size, self.num_tokens),
            dtype=torch.long,
            device=encoder_embeddings.device
        )

        # Step 5: Apply final embedding transformation
        encoder_embeddings = self.embedding_transform(encoder_embeddings)

        # Step 6: Pass through T5 encoder-decoder
        if labels is not None:
            # For T5, we typically create decoder_input_ids by shifting the labels right
            # and prepending the pad token (which is often the same as the decoder start token)
            decoder_input_ids = self._shift_right(labels.clone())

            # # Create a new tensor where -100 replaces pad_token_id if want
            # # This tells the model to ignore pad tokens in loss calculation
            # labels_masked = labels.clone()
            # labels_masked[labels == self.tokenizer.pad_token_id] = -100

            outputs = self.encoder_decoder(
                inputs_embeds=encoder_embeddings,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                # labels=labels_masked,
                labels=labels,
                return_dict=True
            )
        else:
            # For inference, start with a pad token ID
            decoder_input_ids = torch.full(
                (batch_size, 1),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=encoder_embeddings.device
            )

            outputs = self.encoder_decoder(
                inputs_embeds=encoder_embeddings,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )

        return outputs


    def generate(self, input_ids, max_length, attention_mask=None, num_beams=4, early_stopping=False):
        """ Generate text through the inversion process """
        batch_size = input_ids.shape[0]

        # Create attention mask if none provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Prepare generation kwargs with defaults
        generation_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "early_stopping": early_stopping,
            "decoder_start_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
            "output_scores": False,
        }

        # Step 1-6: Same as forward pass
        logits = self.get_llm_logits(input_ids, attention_mask)
        processed_logits = self.process_logits(logits, batch_size)
        encoder_embeddings = self.map_logits_to_encoder_space(processed_logits)
        encoder_attention_mask = torch.ones(
            (batch_size, self.num_tokens),
            dtype=torch.long,
            device=encoder_embeddings.device
        )
        encoder_embeddings = self.embedding_transform(encoder_embeddings)

        # Generate text using the encoder-decoder
        generated_ids = self.encoder_decoder.generate(
            inputs_embeds=encoder_embeddings,
            attention_mask=encoder_attention_mask,
            **generation_kwargs
        )

        return generated_ids


    def invert_text(self, input_text, max_length):
        """ Convenience method to invert a text input back to its presumed source"""
        # Tokenize the input text
        inputs = self.llm_tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.llm.device)
        attention_mask = inputs["attention_mask"].to(self.llm.device)

        # Generate inverted text
        try:
            generated_ids = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length
            )
            # Decode the generated text
            inverted_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            return inverted_text

        except Exception as e:
            print(f"Error during inversion: {e}")
            logits = self.get_llm_logits(input_ids, attention_mask)
            print(f"Logits shape: {logits.shape}")
            processed_logits = self.process_logits(logits, input_ids.shape[0])
            print(f"Processed logits shape: {processed_logits.shape}")
            print(f"Word embeddings shape: {self.word_embeddings.shape}")
            return "Inversion failed due to error."


    def save_pretrained(self, output_dir):
        """ Save the model to disk """
        os.makedirs(output_dir, exist_ok=True)

        # Save the T5 model
        self.encoder_decoder.save_pretrained(os.path.join(output_dir, "t5_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "t5_tokenizer"))

        # Save the transformation layers
        torch.save(self.embedding_transform.state_dict(),
                os.path.join(output_dir, "embedding_transform.pt"))
        torch.save(self.unigram,
                os.path.join(output_dir, "unigram.pt"))

        # Save the configuration
        config = {
            "t5_model_name": os.path.join(output_dir, "t5_model"),
            "llm_model_name": self.llm.config._name_or_path,
            "bottleneck_dim": self.bottleneck_dim,
            "num_tokens": self.num_tokens,
            "unigram_beta": self.unigram_beta,
        }

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f)


    @classmethod
    def from_pretrained(cls, model_path):
        """ Load a pretrained model from disk """
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)

        # Initialize model with the original LLM but the saved T5
        t5_path = os.path.join(model_path, "t5_model")
        t5_tokenizer_path = os.path.join(model_path, "t5_tokenizer")

        model = cls(
            t5_model_name=t5_path,
            t5_tokenizer_name=t5_tokenizer_path,
            llm_model_name=config["llm_model_name"],
            bottleneck_dim=config["bottleneck_dim"],
            num_tokens=config["num_tokens"],
            unigram_beta=config["unigram_beta"]
        )

        # Load the transformation layers
        model.embedding_transform.load_state_dict(
            torch.load(os.path.join(model_path, "embedding_transform.pt"))
        )
        model.unigram = torch.load(
            os.path.join(model_path, "unigram.pt")
        )

        return model


def generate_random_token_sequences(llm_tokenizer, dataset_size, min_length, max_length):
    """
    Generate random token sequences from the LLM vocabulary
    with lengths evenly distributed between min_length and max_length.

    Args:
        llm_tokenizer: The tokenizer for the LLM
        dataset_size: Total number of sequences to generate
        min_length: Minimum sequence length (in tokens)
        max_length: Maximum sequence length (in tokens)

    Returns:
        List of decoded token sequences as strings
    """
    logger.info(f"Generating {dataset_size} random token sequences...")

    # Get vocabulary size
    vocab_size = llm_tokenizer.vocab_size
    logger.info(f"Vocabulary size: {vocab_size}")

    # Reserve some special tokens
    special_token_ids = set()
    for token in llm_tokenizer.special_tokens_map.values():
        if isinstance(token, str):
            special_token_ids.add(llm_tokenizer.convert_tokens_to_ids(token))
        elif isinstance(token, list):
            for t in token:
                special_token_ids.add(llm_tokenizer.convert_tokens_to_ids(t))

    # Create the token ID list (excluding special tokens)
    valid_token_ids = [i for i in range(vocab_size) if i not in special_token_ids]

    # Generate sequences with evenly distributed lengths
    sequences = []

    # Equal number of examples for each length
    examples_per_length = dataset_size // (max_length - min_length + 1)

    for length in range(min_length, max_length + 1):
        for _ in range(examples_per_length):
            # Sample random token IDs
            token_ids = np.random.choice(valid_token_ids, size=length, replace=True).tolist()

            # Decode to text (and handle potential errors)
            try:
                text = llm_tokenizer.decode(token_ids, skip_special_tokens=True)

                # Make sure it's not empty after decoding
                if text.strip():
                    sequences.append(text)
                else:
                    # If empty, try again with different tokens
                    token_ids = np.random.choice(valid_token_ids, size=length, replace=True).tolist()
                    text = llm_tokenizer.decode(token_ids, skip_special_tokens=True)
                    if text.strip():
                        sequences.append(text)
            except Exception as e:
                logger.warning(f"Error decoding tokens: {e}")
                continue

    # Fill any remaining slots to reach dataset_size
    remaining = dataset_size - len(sequences)
    if remaining > 0:
        for _ in range(remaining):
            length = random.randint(min_length, max_length)
            token_ids = np.random.choice(valid_token_ids, size=length, replace=True).tolist()
            try:
                text = llm_tokenizer.decode(token_ids, skip_special_tokens=True)
                if text.strip():
                    sequences.append(text)
            except:
                pass

    # Shuffle the sequences
    random.shuffle(sequences)

    # Trim to dataset_size
    sequences = sequences[:dataset_size]

    logger.info(f"Generated {len(sequences)} sequences")

    # Log some examples
    logger.info("Example sequences:")
    for i in range(min(5, len(sequences))):
        logger.info(f"  {i+1}. '{sequences[i]}'")

    return sequences


class LLMInversionDataset(Dataset):
    """
    Dataset for LLM inversion training.
    Each sample is a string that will be:
    1. Tokenized and passed through the LLM
    2. The output logits will be used to reconstruct the original input
    """
    def __init__(self, text_samples, llm_tokenizer, t5_tokenizer, max_length):
        self.text_samples = text_samples
        self.llm_tokenizer = llm_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text_samples)

    def __getitem__(self, idx):
        text = self.text_samples[idx]

        # Tokenize for LLM input
        llm_tokens = self.llm_tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Tokenize for T5 target (expected output)
        t5_tokens = self.t5_tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "llm_input_ids": llm_tokens["input_ids"].squeeze(),
            "llm_attention_mask": llm_tokens["attention_mask"].squeeze(),
            "t5_target_ids": t5_tokens["input_ids"].squeeze(),
            "t5_target_attention_mask": t5_tokens["attention_mask"].squeeze(),
            "original_text": text
        }


def create_dataloaders(text_samples, llm_tokenizer, t5_tokenizer, batch_size,
                       max_length, val_split, num_workers):
    """ Create training and validation dataloaders from text samples """
    # Split into train and validation
    val_size = int(len(text_samples) * val_split)
    train_samples = text_samples[val_size:]
    val_samples = text_samples[:val_size]

    # Create datasets
    train_dataset = LLMInversionDataset(
        train_samples, llm_tokenizer, t5_tokenizer, max_length
    )
    val_dataset = LLMInversionDataset(
        val_samples, llm_tokenizer, t5_tokenizer, max_length
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """ Train for one epoch """
    model.train()
    total_loss = 0

    # Process batches with progress bar
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        # Move data to device
        llm_input_ids = batch["llm_input_ids"].to(device)
        llm_attention_mask = batch["llm_attention_mask"].to(device)
        t5_target_ids = batch["t5_target_ids"].to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=llm_input_ids,
            attention_mask=llm_attention_mask,
            labels=t5_target_ids
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()
        scheduler.step()

        # Update progress bar
        progress_bar.set_postfix({"train loss": loss.item()})

        # Log every 100 steps
        if (step + 1) % 100 == 0:
            logger.info(f"Epoch {epoch}, Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(model, dataloader, device, epoch):
    """ Validate the model """
    model.eval()
    total_loss = 0

    with torch.no_grad():

        # Process batches with progress bar
        progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")

        for _, batch in enumerate(progress_bar):
            # Move data to device
            llm_input_ids = batch["llm_input_ids"].to(device)
            llm_attention_mask = batch["llm_attention_mask"].to(device)
            t5_target_ids = batch["t5_target_ids"].to(device)

            # Forward pass
            outputs = model(
                input_ids=llm_input_ids,
                attention_mask=llm_attention_mask,
                labels=t5_target_ids
            )
            loss = outputs.loss
            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({"val loss": loss.item()})

    return total_loss / len(dataloader)


def train_inversion_model(
    model,
    seed=42,
    dataset_size=5000,
    min_seq_length=1,
    max_seq_length=24,
    max_length=32,
    val_split=0.1,
    output_dir="./inversion_model_checkpoints",
    batch_size=64,
    mini_batch_size=64,
    num_epochs=5,
    save_steps=1000,
    warmup_steps=0,
    num_workers=4,
    learning_rate=5e-5,
    weight_decay=0.01,
):

    """ Train the LLM inversion model on a dataset of randomly generated token sequences """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Generate random token sequences for training
    logger.info(f"Generating random token sequences dataset of size {dataset_size}...")
    text_samples = generate_random_token_sequences(
        model.llm_tokenizer,
        dataset_size=dataset_size,
        min_length=min_seq_length,
        max_length=max_seq_length
    )

    # Move model to device
    model = model.to(device)
    model.mini_batch_size = mini_batch_size

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        text_samples=text_samples,
        llm_tokenizer=model.llm_tokenizer,
        t5_tokenizer=model.tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        val_split=val_split,
        num_workers=num_workers
    )

    # Freeze the LLM (it should already be frozen in the model constructor)
    for param in model.llm.parameters():
        param.requires_grad = False

    # Only train the T5 model and transformation layers
    # Get trainable parameters only from T5 and transformation layers
    optimizer_params = [
        {"params": model.encoder_decoder.parameters()},
        {"params": model.embedding_transform.parameters()},
    ]

    # Create optimizer
    optimizer = optim.AdamW(
        optimizer_params,
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Create learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch
        )

        # Validate
        val_loss = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            epoch=epoch
        )

        # Log metrics
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint if it's the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
            model_path = os.path.join(output_dir, "best_model")

            # Save the model
            model.save_pretrained(model_path)
            logger.info(f"Saved best model to {model_path}")

        # Save checkpoint every save_steps
        if (epoch * len(train_loader)) % save_steps < len(train_loader):
            model_path = os.path.join(output_dir, f"checkpoint-{global_step}")
            model.save_pretrained(model_path)
            logger.info(f"Saved checkpoint to {model_path}")

        global_step += len(train_loader)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    logger.info("Training completed!")
    return model

def test_inversion_model(model, test_texts, max_length):
    """ Test the inversion model on some sample texts """
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    results = []
    for text in test_texts:
        # Invert the text
        inverted_text = model.invert_text(text, max_length)

        # Calculate simple similarity (character-level overlap)
        # In a real application, you might want to use more sophisticated metrics
        orig_chars = set(text.lower())
        inv_chars = set(inverted_text.lower())
        if len(orig_chars) > 0:
            overlap = len(orig_chars.intersection(inv_chars)) / len(orig_chars)
        else:
            overlap = 0

        results.append({
            "original": text,
            "inverted": inverted_text,
            "char_overlap": overlap
        })
        print(f"Original: {text}")
        print(f"Inverted: {inverted_text}")
        print(f"Character overlap: {overlap:.2f}")
        print("---")

    return results

def test_inversion_model_dataset(model, loaded_text_samples, eval_batch_size, input_len, max_length):
    """ Test inversion model on batch of llm input tensors """
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Create the evaluation dataset and dataloader
    eval_dataset = LLMInversionDataset(text_samples=loaded_text_samples, llm_tokenizer=model.llm_tokenizer,
        t5_tokenizer=model.tokenizer, max_length=max_length)
    eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=12, pin_memory=True)

    results = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Batch Inverting Tokens"):
            # Batch generate inverted texts
            generated_ids_batch = model.generate(
                input_ids=batch["llm_input_ids"].to(device),
                attention_mask=batch["llm_attention_mask"].to(device),
                max_length=max_length,
            )

            # Decode the generated IDs and original texts (get rid of pad token and possibly space)
            # Unlike LLM decoder, T5 decoder gets rid of trailing white space so have to add back in to evaluate fairly
            pred_texts_batch = model.tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=False)
            pred_texts_batch = [text[5:] if batch["original_text"][t][0]==" " else text[6:] for t,text in enumerate(pred_texts_batch)]

            pred_tokens_batch = model.llm_tokenizer(pred_texts_batch).input_ids
            true_tokens_batch = batch["llm_input_ids"]

            for i in range(len(pred_tokens_batch)):
                pred_tokens_batch_i = pred_tokens_batch[i][:input_len]
                while len(pred_tokens_batch_i) < input_len:
                    pred_tokens_batch_i += [model.tokenizer.pad_token_id]
                results.append({
                    "true_tokens": true_tokens_batch[i][:input_len].to("cpu"),
                    "pred_tokens": torch.tensor(pred_tokens_batch_i).to("cpu"),
                    "found_solution": True,
                    "done_epochs": 1,
                })

    stats = get_paper_summary_stats_new(results, epochs=1)
    print("\nexact_inversion", stats["percent_exact_inversion"])

    scores = []
    # Get partial match score
    for result in results:
        score = 0
        pred = result["pred_tokens"].tolist()
        true = result["true_tokens"].tolist()
        for i in range(len(pred)):
            if pred[i] == true[i]:
                score += 1
        scores.append((score/len(pred))*100)
    stats["percent_exact_inversion"] = np.mean(scores)
    stats["percent_exact_inversion_ste"] = np.std(scores, ddof=1) / np.sqrt(len(scores))
    print("\npartial_inversion", stats["percent_exact_inversion"], "+-", stats["percent_exact_inversion_ste"])

    return results, stats