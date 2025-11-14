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
from datasets import load_dataset
import os
import json
import re
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
    This model takes generated tokens from a causal LM and predicts the original input
    that would have generated those tokens using T5's pretrained token embeddings.
    """
    def __init__(
        self,
        t5_model_name="t5-base",
        t5_tokenizer_name="t5-base",
        llm_model_name="meta-llama/Llama-2-7b-hf",
        num_generation_tokens=25,
    ):
        super().__init__()

        # Load base models
        self.encoder_decoder = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)

        # Load tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(t5_tokenizer_name)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # Freeze the LLM
        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm.eval()

        # Configuration
        self.num_generation_tokens = num_generation_tokens

    def generate_llm_tokens(self, input_ids, attention_mask):
        """Generate tokens from the LLM model"""
        with torch.no_grad():
            generated = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.num_generation_tokens,
                do_sample=False,  # Use greedy decoding for consistency
                pad_token_id=self.llm_tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_hidden_states=False,
            )

            # Extract only the newly generated tokens (excluding the input)
            input_length = input_ids.shape[1]
            generated_tokens = generated.sequences[:, input_length:]

        return generated_tokens

    def get_t5_tokens_from_llm_output(self, llm_tokens):
        """Convert LLAMA generated tokens to T5 token IDs"""
        batch_size = llm_tokens.shape[0]
        t5_input_ids_list = []
        t5_attention_masks_list = []

        for i in range(batch_size):
            # Decode LLAMA tokens to text
            llm_text = self.llm_tokenizer.decode(llm_tokens[i], skip_special_tokens=True)

            # Re-tokenize with T5 tokenizer
            t5_inputs = self.tokenizer(
                llm_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Reasonable max length for T5
            )

            t5_input_ids_list.append(t5_inputs["input_ids"].squeeze(0))
            t5_attention_masks_list.append(t5_inputs["attention_mask"].squeeze(0))

        # Pad sequences to same length
        max_length = max(seq.shape[0] for seq in t5_input_ids_list)

        padded_input_ids = []
        padded_attention_masks = []

        for input_ids, attention_mask in zip(t5_input_ids_list, t5_attention_masks_list):
            pad_length = max_length - input_ids.shape[0]
            if pad_length > 0:
                padded_input_ids.append(torch.cat([
                    input_ids,
                    torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
                ]))
                padded_attention_masks.append(torch.cat([
                    attention_mask,
                    torch.zeros(pad_length, dtype=attention_mask.dtype)
                ]))
            else:
                padded_input_ids.append(input_ids)
                padded_attention_masks.append(attention_mask)

        return (
            torch.stack(padded_input_ids).to(llm_tokens.device),
            torch.stack(padded_attention_masks).to(llm_tokens.device)
        )

    def convert_to_t5_tokens(self, llm_tokens):
        """Legacy method name - use get_t5_tokens_from_llm_output instead"""
        return self.get_t5_tokens_from_llm_output(llm_tokens)

    def _shift_right(self, input_ids):
        """Shift input ids one token to the right for T5 decoder input"""
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.tokenizer.pad_token_id
        return shifted_input_ids

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass of the token inversion model"""
        # Create attention mask if none provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Step 1: Generate tokens from LLM
        generated_tokens = self.generate_llm_tokens(input_ids, attention_mask)

        # Step 2: Convert LLAMA tokens to T5 tokens
        t5_input_ids, t5_attention_mask = self.get_t5_tokens_from_llm_output(generated_tokens)

        # Step 3: Pass through T5 encoder-decoder using T5's own token embeddings
        if labels is not None:
            # For training, we have target labels
            decoder_input_ids = self._shift_right(labels.clone())

            outputs = self.encoder_decoder(
                input_ids=t5_input_ids,
                attention_mask=t5_attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )
        else:
            # For inference
            outputs = self.encoder_decoder(
                input_ids=t5_input_ids,
                attention_mask=t5_attention_mask,
                return_dict=True
            )

        return outputs

    def generate(self, input_ids, max_length, attention_mask=None, num_beams=4, early_stopping=False):
        """Generate text through the token inversion process"""
        # Create attention mask if none provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Prepare generation kwargs
        generation_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "early_stopping": early_stopping,
            "decoder_start_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }

        # Generate tokens and convert to T5 format
        generated_tokens = self.generate_llm_tokens(input_ids, attention_mask)
        t5_input_ids, t5_attention_mask = self.get_t5_tokens_from_llm_output(generated_tokens)

        # Generate inverted text using T5
        generated_ids = self.encoder_decoder.generate(
            input_ids=t5_input_ids,
            attention_mask=t5_attention_mask,
            **generation_kwargs
        )

        return generated_ids

    def invert_text(self, input_text, max_length):
        """Convenience method to invert a text input back to its presumed source"""
        # Tokenize the input text
        inputs = self.llm_tokenizer(input_text, return_tensors="pt").to(self.llm.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Generate inverted text
        generated_ids = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length
        )
        # Decode the generated text
        inverted_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return inverted_text

    def save_pretrained(self, output_dir):
        """Save the model to disk"""
        os.makedirs(output_dir, exist_ok=True)

        # Save the T5 model
        self.encoder_decoder.save_pretrained(os.path.join(output_dir, "t5_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "t5_tokenizer"))

        # Save the configuration
        config = {
            "t5_model_name": os.path.join(output_dir, "t5_model"),
            "llm_model_name": self.llm.config._name_or_path,
            "num_generation_tokens": self.num_generation_tokens,
        }

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, model_path):
        """Load a pretrained model from disk"""
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)

        # Initialize model
        t5_path = os.path.join(model_path, "t5_model")
        t5_tokenizer_path = os.path.join(model_path, "t5_tokenizer")

        model = cls(
            t5_model_name=t5_path,
            t5_tokenizer_name=t5_tokenizer_path,
            llm_model_name=config["llm_model_name"],
            num_generation_tokens=config["num_generation_tokens"]
        )

        return model


def load_variable_length_dataset(llm_tokenizer, target_strategy, dataset_size, min_length, max_length, include_bos=False, random_sentence=True):
    """
    Load text sequences from HuggingFace datasets with variable lengths,
    ensuring different data for each length bin.

    Args:
        llm_tokenizer: The tokenizer for the LLM
        target_strategy: Dataset to use ("tinystories", "reddit")
        dataset_size: Total number of sequences to generate
        min_length: Minimum sequence length (in tokens)
        max_length: Maximum sequence length (in tokens)
        include_bos: Whether to include beginning-of-sequence token
        random_sentence: Whether to randomly select sentences from text

    Returns:
        List of decoded token sequences as strings
    """

    logger.info(f"Loading {dataset_size} variable length sequences from {target_strategy}...")

    # Dataset configuration
    name, split, ind = {
        "tinystories": ["roneneldan/TinyStories", "train", "text"],
        "reddit": ["sentence-transformers/reddit", "train", "body"],
    }[target_strategy]

    ds = load_dataset(name, split=split, streaming=True)

    # Calculate examples per length (evenly distributed)
    examples_per_length = dataset_size // (max_length - min_length + 1)

    sequences = []
    length_counts = {length: 0 for length in range(min_length, max_length + 1)}

    # Keep track of processed data to ensure different data for each length
    dataset_counter = 0

    for data in ds:
        dataset_counter += 1

        # Skip first 1000 samples (reserved for evaluation)
        if dataset_counter <= 1000:
            continue

        # Break if we have enough sequences
        if len(sequences) >= dataset_size:
            break

        try:
            # Get the text content
            string = data[ind]
            if len(string) > 2000:  # Limit very long texts
                string = string[:2000]

            # Choose which sentence to take if random_sentence is True
            if random_sentence:
                sentence_pattern = r'(?<=[.!?])\s+'
                string_list = re.split(sentence_pattern, string)
                string_list = [s.strip() for s in string_list if s.strip()]  # Remove empty strings
                if string_list:
                    string = random.choice(string_list)

            # Tokenize the string
            tokens = llm_tokenizer.encode(string)

            # Skip if text is too short
            if len(tokens) < min_length:
                continue

            # Try to find a length bin that still needs examples
            possible_lengths = []
            for length in range(min_length, max_length + 1):
                if length_counts[length] < examples_per_length and len(tokens) >= length:
                    possible_lengths.append(length)

            # If no lengths need more examples, check if we need to fill remaining slots
            if not possible_lengths:
                remaining_total = dataset_size - len(sequences)
                if remaining_total > 0:
                    # Pick any valid length for remaining slots
                    possible_lengths = [length for length in range(min_length, max_length + 1)
                                     if len(tokens) >= length]

            if not possible_lengths:
                continue

            # Choose a target length
            target_length = random.choice(possible_lengths)

            # Extract tokens of target length
            offset = 0 if include_bos else (1 if tokens[0] == llm_tokenizer.bos_token_id else 0)

            # Random start position
            max_start = len(tokens) - target_length - offset
            if max_start > 0:
                start_pos = offset + random.randint(0, max_start)
            else:
                start_pos = offset

            selected_tokens = tokens[start_pos:start_pos + target_length]

            # Ensure we have the right length
            if len(selected_tokens) == target_length:
                # Decode back to text
                text = llm_tokenizer.decode(selected_tokens, skip_special_tokens=True)

                # Make sure it's not empty after decoding
                if text.strip():
                    sequences.append(text.strip())
                    if target_length in length_counts:
                        length_counts[target_length] += 1

        except Exception as e:
            logger.warning(f"Error processing data point {dataset_counter}: {e}")
            continue

    # Shuffle the sequences to mix different lengths
    random.shuffle(sequences)

    # Trim to exact dataset_size
    sequences = sequences[:dataset_size]

    logger.info(f"Generated {len(sequences)} sequences")
    logger.info(f"Length distribution: {dict(sorted(length_counts.items()))}")

    return sequences

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
    # progress_bar = dataloader

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
        # progress_bar = dataloader

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
    dataset="random"
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

    # Generate sequences for training
    logger.info(f"Generating sequences dataset of size {dataset_size}...")
    if dataset == "random":
        text_samples = generate_random_token_sequences(
            model.llm_tokenizer,
            dataset_size=dataset_size,
            min_length=min_seq_length,
            max_length=max_seq_length
        )
    else:
        text_samples = load_variable_length_dataset(
            model.llm_tokenizer,
            target_strategy=dataset,
            dataset_size=dataset_size,
            min_length=min_seq_length,
            max_length=max_seq_length,
            include_bos=False,
            random_sentence=True
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
        print(f"Starting Training epoch {epoch}")

        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch
        )
        print(f"Training loss {train_loss}")


        # Validate
        val_loss = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            epoch=epoch
        )
        print(f"Val loss {val_loss}")

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
        print(f"Inverted: {[inverted_text]}")
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
        # for batch in eval_dataloader:
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
    print("partial_inversion", stats["percent_exact_inversion"], "+-", stats["percent_exact_inversion_ste"])
    print()

    return results, stats