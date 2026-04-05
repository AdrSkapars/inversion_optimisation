import torch
import random
import re
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
import os
import logging

logger = logging.getLogger(__name__)


def generate_training_pairs(
    model, dataset_name, num_examples,
    min_input_len=1, max_input_len=10,
    num_output_tokens=25, device="cuda",
    seed=42, batch_size=256,
):
    """
    Generate (input_text, output_text) training pairs for reverse model LoRA.

    For each example:
    1. Sample/load input tokens (1-10 tokens)
    2. Run through the forward LLM to get 25 greedy output tokens
    3. Decode both to text strings

    Args:
        model: TransformerLens HookedTransformer (the forward model being inverted)
        dataset_name: "random", "tinystories", or "reddit"
        num_examples: Total number of training pairs to generate
        min_input_len: Minimum input length in tokens
        max_input_len: Maximum input length in tokens
        num_output_tokens: Number of output tokens to generate (25)
        device: Device string
        seed: Random seed
        batch_size: Batch size for generation

    Returns:
        List of (input_text, output_text) tuples
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    examples_per_length = num_examples // (max_input_len - min_input_len + 1)
    pairs = []

    if dataset_name == "random":
        vocab_size = len(model.tokenizer.vocab)
        for input_len in tqdm(range(min_input_len, max_input_len + 1), desc="Generating random pairs"):
            for batch_start in range(0, examples_per_length, batch_size):
                batch_count = min(batch_size, examples_per_length - batch_start)
                tokens = torch.randint(0, vocab_size, (batch_count, input_len)).to(device)
                output_tokens = model.generate(
                    tokens, max_new_tokens=num_output_tokens,
                    do_sample=False, stop_at_eos=False, verbose=False, return_type="tokens",
                )
                for i in range(batch_count):
                    input_text = model.tokenizer.decode(tokens[i].tolist())
                    output_text = model.tokenizer.decode(output_tokens[i, input_len:].tolist())
                    pairs.append((input_text, output_text))

    else:
        # tinystories or reddit
        name, split, ind = {
            "tinystories": ("roneneldan/TinyStories", "train", "text"),
            "reddit": ("sentence-transformers/reddit", "train", "body"),
        }[dataset_name]

        ds = load_dataset(name, split=split, streaming=True)

        # For reddit, skip first 1000 (reserved for eval)
        skip_n = 1000 if dataset_name == "reddit" else 0

        # Collect tokens for each input length
        length_buckets = {l: [] for l in range(min_input_len, max_input_len + 1)}

        dataset_counter = 0
        for data in tqdm(ds, desc=f"Loading {dataset_name} data"):
            dataset_counter += 1
            if dataset_counter <= skip_n:
                continue

            if all(len(v) >= examples_per_length for v in length_buckets.values()):
                break

            try:
                string = data[ind][:2000]
                sentence_pattern = r'(?<=[.!?])\s+'
                sentences = re.split(sentence_pattern, string)
                sentences = [s.strip() for s in sentences if s.strip()]
                if not sentences:
                    continue
                string = random.choice(sentences)

                tokens = model.to_tokens(string)[0]
                offset = 1 if tokens[0] == model.tokenizer.bos_token_id else 0
                tokens = tokens[offset:]

                if len(tokens) < min_input_len:
                    continue

                possible = [l for l in range(min_input_len, max_input_len + 1)
                            if len(length_buckets[l]) < examples_per_length and len(tokens) >= l]
                if not possible:
                    continue

                target_len = random.choice(possible)
                max_start = len(tokens) - target_len
                start = random.randint(0, max_start) if max_start > 0 else 0
                selected = tokens[start:start + target_len].unsqueeze(0)

                length_buckets[target_len].append(selected)

            except Exception:
                continue

        # Generate outputs in batches per input length
        for input_len in tqdm(range(min_input_len, max_input_len + 1), desc="Generating outputs"):
            bucket_tokens = length_buckets[input_len][:examples_per_length]
            if not bucket_tokens:
                logger.warning(f"No tokens collected for input_len={input_len}")
                continue
            all_tokens = torch.cat(bucket_tokens, dim=0).to(device)

            for batch_start in range(0, len(all_tokens), batch_size):
                batch = all_tokens[batch_start:batch_start + batch_size]
                output_tokens = model.generate(
                    batch, max_new_tokens=num_output_tokens,
                    do_sample=False, stop_at_eos=False, verbose=False, return_type="tokens",
                )
                for i in range(len(batch)):
                    input_text = model.tokenizer.decode(batch[i].tolist())
                    output_text = model.tokenizer.decode(output_tokens[i, input_len:].tolist())
                    pairs.append((input_text, output_text))

    random.shuffle(pairs)
    return pairs[:num_examples]


class ReverseModelSFTDataset(Dataset):
    """
    Dataset that formats (input_text, output_text) pairs for reverse model SFT.

    Each example becomes:
    - input_ids: reversed(output_tokens) + reversed(input_tokens)  [BOS dropped]
    - labels: [-100 for prompt portion] + reversed(input_tokens)
    """

    def __init__(self, text_pairs, tokenizer, max_length=128):
        self.text_pairs = text_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        input_text, output_text = self.text_pairs[idx]

        # Tokenize with the reverse model's tokenizer
        output_ids = self.tokenizer.encode(output_text)
        input_ids = self.tokenizer.encode(input_text)

        # Reverse and drop BOS (which ends up at the end after reversing)
        reversed_output = list(reversed(output_ids))[:-1]
        reversed_input = list(reversed(input_ids))[:-1]

        # Full sequence: reversed output (prompt) + reversed input (completion)
        full_ids = reversed_output + reversed_input
        prompt_len = len(reversed_output)

        # Labels: -100 for prompt, actual IDs for completion
        labels = [-100] * prompt_len + reversed_input

        # Truncate if needed
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        }


def _collate_fn(batch, pad_token_id):
    """Pad batch to same length."""
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for b in batch:
        pad_len = max_len - b["input_ids"].size(0)
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def train_reverse_model_lora(
    text_pairs,
    model_id="Corning/Reverse-Model-7B-348B",
    output_dir="data/ReverseModelLoRA",
    lora_rank=32,
    lora_alpha=64,
    lora_dropout=0.05,
    num_epochs=2,
    batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    max_length=128,
    val_split=0.05,
    logging_steps=50,
    max_steps=-1,
    seed=42,
):
    """
    Train a LoRA adapter on the reverse model for inversion.

    Args:
        text_pairs: List of (input_text, output_text) tuples from generate_training_pairs
        model_id: HuggingFace model ID for the base reverse model
        output_dir: Directory to save checkpoints and final adapter
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio
        max_length: Max sequence length for training
        val_split: Fraction of data for validation
        logging_steps: Log training loss every N steps
        max_steps: Stop after N steps (-1 for full epochs)
        seed: Random seed

    Returns:
        (model, tokenizer) — the LoRA model and tokenizer
    """
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    rev_model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    rev_model = get_peft_model(rev_model, lora_config)
    rev_model.print_trainable_parameters()

    # Split into train/val
    random.seed(seed)
    shuffled_pairs = text_pairs.copy()
    random.shuffle(shuffled_pairs)
    val_size = int(len(shuffled_pairs) * val_split)
    val_pairs = shuffled_pairs[:val_size]
    train_pairs = shuffled_pairs[val_size:]

    train_dataset = ReverseModelSFTDataset(train_pairs, tokenizer, max_length)
    val_dataset = ReverseModelSFTDataset(val_pairs, tokenizer, max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        bf16=True,
        logging_steps=logging_steps,
        eval_strategy="epoch" if max_steps == -1 else "no",
        save_strategy="epoch" if max_steps == -1 else "no",
        save_total_limit=2,
        load_best_model_at_end=True if max_steps == -1 else False,
        metric_for_best_model="eval_loss",
        seed=seed,
        report_to="none",
    )

    pad_id = tokenizer.pad_token_id
    trainer = Trainer(
        model=rev_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda batch: _collate_fn(batch, pad_id),
    )

    trainer.train()

    # Save final LoRA adapter
    best_model_dir = os.path.join(output_dir, "best_lora")
    rev_model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    return rev_model, tokenizer
