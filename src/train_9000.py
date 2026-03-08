import os
import torch
from datasets import load_dataset, DatasetDict
from sklearn.metrics import roc_auc_score, log_loss
from transformers import (ElectraTokenizer, 
                          ElectraForSequenceClassification, 
                          TrainingArguments, Trainer,
                          logging as hf_logging,
                          ProgressCallback,
)

# full_data_9000
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

    MODEL_NAME = "monologg/koelectra-base-v3-discriminator"  
    tokenizer  = ElectraTokenizer.from_pretrained(MODEL_NAME)
    model = ElectraForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2         
    )

    dataset = load_dataset("../tokenized_ds") # save_ds로 변환된 train_valid 데이터셋

    def add_token_length(batch):
        batch["token_length"] = [sum(mask) for mask in batch["attention_mask"]]
        return batch

    dataset = dataset.map(add_token_length, batched=True, num_proc=30)

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label", "token_length"]
    )

    def filter_not_truncated(example):
        return example["token_length"] < 512
        
    filtered_ds = DatasetDict({ 
        "train": dataset["train"].filter(filter_not_truncated),
        "valid": dataset["validation"].filter(filter_not_truncated),
    })

    hf_logging.set_verbosity_info()
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()

    device = torch.device("cuda:0")  # 내부적으로는 0번으로 보입니다
    model.to(device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        return {
            "auc": roc_auc_score(labels, probs),
            "log_loss": log_loss(labels, probs),
        }
        
    args = TrainingArguments(
        output_dir="koelectra_detector",
        per_device_train_batch_size=78//2,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        logging_steps=200,
        logging_first_step=True,
        report_to=[],
        log_level="info",
        learning_rate=2e-5,
        num_train_epochs=10,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=filtered_ds["train"],
        eval_dataset=filtered_ds["valid"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[ProgressCallback],
    )

    trainer.train()

if __name__ == "__main__":
    main()
