
import os
import torch
import random
import numpy as np
import argparse
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, log_loss
from transformers import (
    ElectraTokenizer,
    ElectraForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
    set_seed
)
from torch.utils.data import DataLoader, Sampler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
import logging
from dataclasses import dataclass, field

# --- Custom Model ---
from modeling import SuperLossElectraForSequenceClassification

# --- Logger Setup ---
def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "training.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    logger.propagate = False
    return logger

logger = setup_logger()

@dataclass
class TrainingConfig:
    model_name: str = "monologg/koelectra-base-v3-discriminator"
    output_dir: str = "models/koelectra-detector"
    train_batch_size: int = 78
    eval_batch_size: int = 488
    gradient_accumulation_steps: int = 2
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 200
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    best_metric: str = "auc"
    load_best_model_at_end: bool = True
    downsample_ratio: float = 0.4
    seed: int = 42
    use_superloss: bool = False
    processed_data_path: str = "processed_data/tokenized_ds"

class BalancedSampler(Sampler):
    def __init__(self, dataset, downsample_ratio):
        self.dataset = dataset
        self.downsample_ratio = downsample_ratio
        labels = np.array(dataset['label'])
        self.indices_label_0 = np.where(labels == 0)[0]
        self.indices_label_1 = np.where(labels == 1)[0]
        self.num_samples_label_0 = int(len(self.indices_label_0) * self.downsample_ratio)
        self.num_samples = self.num_samples_label_0 + len(self.indices_label_1)
        logger.info(f"Sampler created: Using all {len(self.indices_label_1)} samples of label 1 and {self.num_samples_label_0} ({self.downsample_ratio*100}%) of samples of label 0.")

    def __iter__(self):
        sampled_indices_label_0 = np.random.choice(self.indices_label_0, self.num_samples_label_0, replace=False)
        combined_indices = np.concatenate([sampled_indices_label_0, self.indices_label_1])
        np.random.shuffle(combined_indices)
        return iter(combined_indices.tolist())

    def __len__(self):
        return self.num_samples

class DatasetManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = ElectraTokenizer.from_pretrained(config.model_name)
        self.dataset = None
        self.filtered_ds = None

    def load_and_prepare_dataset(self):
        self.dataset = load_dataset(self.config.processed_data_path)
        self.dataset = self.dataset.map(self._add_token_length, batched=True, num_proc=30)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "token_length"])
        self.filtered_ds = self.dataset.filter(self._filter_not_truncated)
        logger.info("Dataset loaded and prepared.")

    def _add_token_length(self, batch):
        batch["token_length"] = [sum(mask) for mask in batch["attention_mask"]]
        return batch

    def _filter_not_truncated(self, example):
        return example["token_length"] < 512

    def get_dataloaders(self):
        if self.filtered_ds is None:
            self.load_and_prepare_dataset()
        train_sampler = BalancedSampler(self.filtered_ds["train"], self.config.downsample_ratio)
        train_dataloader = DataLoader(self.filtered_ds["train"], batch_size=self.config.train_batch_size, sampler=train_sampler, num_workers=4)
        eval_dataloader = DataLoader(self.filtered_ds["test"], batch_size=self.config.eval_batch_size, num_workers=4)
        return train_dataloader, eval_dataloader, self.tokenizer

class CustomTrainer:
    def __init__(self, config: TrainingConfig, model, tokenizer, train_dataloader, eval_dataloader):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.scaler = GradScaler()
        num_update_steps_per_epoch = -(-len(self.train_dataloader) // self.config.gradient_accumulation_steps)
        self.num_training_steps = self.config.num_train_epochs * num_update_steps_per_epoch
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=self.num_training_steps)

    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        return {"auc": roc_auc_score(labels, probs), "log_loss": log_loss(labels, probs)}

    def evaluate(self, dataloader, description="Evaluation"):
        self.model.eval()
        all_logits = []
        all_labels = []
        with torch.inference_mode(), autocast(self.device.type, torch.bfloat16):
            for batch in tqdm(dataloader, desc=description, dynamic_ncols=True):
                batch_on_device = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch_on_device.pop("label")
                idx = max(batch_on_device["attention_mask"].sum(-1))
                outputs = self.model(input_ids=batch_on_device["input_ids"][:, :idx], attention_mask=batch_on_device["attention_mask"][:, :idx])
                all_logits.append(outputs.logits.cpu().float())
                all_labels.append(labels.cpu().float())
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        metrics = self._compute_metrics((all_logits.numpy(), all_labels.numpy()))
        self.model.train()
        return metrics

    def train(self):
        global_step = 0
        best_metric_value = -1.0
        progress_bar = tqdm(range(self.num_training_steps), dynamic_ncols=True)

        for epoch in range(self.config.num_train_epochs):
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                idx = max(batch["attention_mask"].sum(-1))
                with autocast(self.device.type, torch.bfloat16):
                    outputs = self.model(input_ids=batch["input_ids"][:, :idx], attention_mask=batch["attention_mask"][:, :idx], labels=batch["label"])
                    loss = outputs.loss
                    loss = loss / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    progress_bar.update(1)
                    if global_step % self.config.logging_steps == 0:
                        logger.info(f"Epoch {epoch+1}, Step {global_step}: Train loss: {loss.item() * self.config.gradient_accumulation_steps:.4f}")
                    if global_step % self.config.eval_steps == 0:
                        metrics = self.evaluate(self.eval_dataloader, f"Eval at step {global_step}")
                        logger.info(f"Epoch {epoch+1}, Step {global_step}: Eval metrics: {metrics}")
                        if metrics[self.config.best_metric] > best_metric_value:
                            best_metric_value = metrics[self.config.best_metric]
                            logger.info(f"New best model found with {self.config.best_metric}: {best_metric_value:.4f}")
                            best_model_path = os.path.join(self.config.output_dir, "best_model")
                            self.model.save_pretrained(best_model_path)
                            self.tokenizer.save_pretrained(best_model_path)
                    if global_step % self.config.save_steps == 0:
                        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{global_step}")
                        self.model.save_pretrained(checkpoint_dir)
                        self.tokenizer.save_pretrained(checkpoint_dir)
                        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        if self.config.load_best_model_at_end:
            logger.info("Loading best model for final evaluation...")
            best_model_path = os.path.join(self.config.output_dir, "best_model")
            self.model = ElectraForSequenceClassification.from_pretrained(best_model_path)
            self.model.to(self.device)
            final_metrics = self.evaluate(self.eval_dataloader, "Final Evaluation")
            logger.info(f"Final evaluation metrics: {final_metrics}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_superloss", action="store_true", help="Use SuperLoss for training.")
    parser.add_argument("--output_dir", type=str, default="models/koelectra-detector", help="Output directory for the trained model.")
    parser.add_argument("--processed_data_path", type=str, default="processed_data/tokenized_ds", help="Path to the processed dataset.")
    args = parser.parse_args()

    config = TrainingConfig(use_superloss=args.use_superloss, output_dir=args.output_dir, processed_data_path=args.processed_data_path)

    set_seed(config.seed)

    dataset_manager = DatasetManager(config)
    train_dataloader, eval_dataloader, tokenizer = dataset_manager.get_dataloaders()

    if config.use_superloss:
        model = SuperLossElectraForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    else:
        model = ElectraForSequenceClassification.from_pretrained(config.model_name, num_labels=2)

    trainer = CustomTrainer(config, model, tokenizer, train_dataloader, eval_dataloader)
    trainer.train()

if __name__ == "__main__":
    main()
