import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import ElectraTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from superloss import HardFirstSuperLoss
from sklearn.metrics import roc_auc_score, log_loss

# --- Data Preparation ---
df = pd.read_csv("data/train.csv")
df['full_text'] = df['full_text'].str.split('\n')
df = (
    df[['title','full_text', 'generated']]
    .explode('full_text')
    .rename(columns={'full_text': 'full_text'})
    .reset_index(drop=True)
)
df = df.rename(columns={"generated":"label"})
df['char_len'] = df['full_text'].str.len()
top183_idx = df.nlargest(400, 'char_len').index
df = df.drop(index=top183_idx).reset_index(drop=True)

ds = Dataset.from_pandas(df[["full_text", "label"]])
ds = ds.cast_column("label", ClassLabel(names=["human", "ai"]))
ds_split = ds.train_test_split(test_size=0.1, stratify_by_column="label", seed=42)
dataset = DatasetDict({"train": ds_split["train"], "eval":  ds_split["test"]})

# --- Tokenization ---
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
tokenizer  = ElectraTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    return tokenizer(
        batch["full_text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

dataset = dataset.map(tokenize_batch, batched=True, remove_columns=["full_text"], num_proc=20)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dataset.save_to_disk("processed_data/Ada")

# --- Custom Trainer ---
class SuperLossTrainer(Trainer):
    hfs = HardFirstSuperLoss(2, 0.7, 0.1)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        base_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='none'
        )
        spl, sig = self.hfs(base_loss)
        super_loss = spl.mean()
        return (super_loss, outputs) if return_outputs else super_loss

# --- Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return {
        "auc": roc_auc_score(labels, probs),
        "log_loss": log_loss(labels, probs),
    }

# --- Training ---
model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

args = TrainingArguments(
    output_dir="models/koelectra-detector_Ada",
    per_device_train_batch_size=78,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_steps=200,
    learning_rate=2e-5,
    num_train_epochs=10,
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    bf16=True,
)

trainer = SuperLossTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
