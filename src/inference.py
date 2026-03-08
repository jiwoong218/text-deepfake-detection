
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from tqdm import tqdm

def create_overlapping_windows(text, tokenizer, max_length=512, overlap=256):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_length - 2:
        return [text]

    windows = []
    start = 0
    stride = max_length - overlap - 2
    while start < len(tokens):
        end = min(start + max_length - 2, len(tokens))
        window_tokens = tokens[start:end]
        window_text = tokenizer.convert_tokens_to_string(window_tokens)
        windows.append(window_text)
        if end >= len(tokens):
            break
        start += stride
    return windows

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint path not found at '{args.checkpoint}'")
        return

    tokenizer = ElectraTokenizer.from_pretrained(args.model_name)
    model = ElectraForSequenceClassification.from_pretrained(args.checkpoint).to(device).eval()
    print("Model and tokenizer loaded successfully.")

    try:
        test_df = pd.read_csv(args.test_csv)
        test_df = test_df.rename(columns={"paragraph_text": "full_text"})
    except FileNotFoundError:
        print(f"Error: '{args.test_csv}' not found.")
        return

    if args.use_overlapping_windows:
        all_windows = []
        window_to_original_idx = []
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing Texts"):
            text = row['full_text']
            windows = create_overlapping_windows(text, tokenizer, max_length=args.max_length, overlap=args.overlap)
            all_windows.extend(windows)
            window_to_original_idx.extend([idx] * len(windows))
        window_dataset = Dataset.from_dict({"text": all_windows})
    else:
        window_dataset = Dataset.from_pandas(test_df[["full_text"]].rename(columns={"full_text": "text"}))

    def tok_fn(batch):
        return tokenizer(batch["text"], padding="longest", truncation=True, max_length=args.max_length)

    window_dataset = window_dataset.map(tok_fn, batched=True, remove_columns=["text"])
    window_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    test_loader = DataLoader(window_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    window_probs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running Inference"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            window_probs.extend(probs)

    if args.use_overlapping_windows:
        final_probs = []
        for original_idx in range(len(test_df)):
            window_indices = [i for i, orig_idx in enumerate(window_to_original_idx) if orig_idx == original_idx]
            window_probs_for_text = [window_probs[i] for i in window_indices]
            max_prob = max(window_probs_for_text) if window_probs_for_text else 0.0
            final_probs.append(max_prob)
    else:
        final_probs = window_probs

    try:
        submission_df = pd.read_csv(args.submission_csv)
        submission_df["generated"] = final_probs
        submission_df.to_csv(f"{args.output}.csv", index=False)
        print(f"✅ '{args.output}.csv' saved successfully!")
    except FileNotFoundError:
        print(f"Error: '{args.submission_csv}' not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output CSV name (without extension)")
    parser.add_argument("--test_csv", type=str, default="data/test.csv", help="Path to the test CSV file.")
    parser.add_argument("--submission_csv", type=str, default="data/sample_submission.csv", help="Path to the sample submission CSV file.")
    parser.add_argument("--model_name", type=str, default="monologg/koelectra-base-v3-discriminator", help="Model name for the tokenizer.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--use_overlapping_windows", action="store_true", help="Use overlapping windows for long texts.")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap size for overlapping windows.")
    args = parser.parse_args()
    main(args)
