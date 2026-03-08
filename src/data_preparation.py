
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import ElectraTokenizer
import argparse
import os

def prepare_dataset(args):
    df = pd.read_csv(args.input_csv)
    df['full_text'] = (df['full_text']
                       .str.split('\n')
                       .apply(lambda lst: [s.strip() for s in lst])
    )
    df = (
        df[['title', 'full_text', 'generated']]
        .explode('full_text')
        .reset_index(drop=True)
    )
    df = df.rename(columns={"generated":"label"})

    ds = Dataset.from_pandas(df[["full_text", "label"]])
    ds = ds.cast_column("label", ClassLabel(names=["human", "ai"]))
    ds_split = ds.train_test_split(test_size=args.test_size, stratify_by_column="label", seed=args.seed)

    dataset = DatasetDict({"train": ds_split["train"], "eval": ds_split["test"]})

    tokenizer = ElectraTokenizer.from_pretrained(args.model_name)

    def tokenize_batch(batch):
        return tokenizer(
            batch["full_text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    dataset = dataset.map(tokenize_batch, batched=True, remove_columns=["full_text"], num_proc=30)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(os.path.join(args.output_dir, args.dataset_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data/train.csv", help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Directory to save the processed dataset.")
    parser.add_argument("--dataset_name", type=str, default="tokenized_ds", help="Name of the processed dataset.")
    parser.add_argument("--model_name", type=str, default="monologg/koelectra-base-v3-discriminator", help="Model name for the tokenizer.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train_test_split.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization.")
    args = parser.parse_args()
    prepare_dataset(args)
