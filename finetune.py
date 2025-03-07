import torch
import json
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import evaluate
from peft import get_peft_model, LoraConfig, TaskType

class TravelLLMTrainer:
    def __init__(self, model_name="microsoft/phi-1_5", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        # Configure LoRA for better fine-tuning
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Increased for better learning
            lora_alpha=32,
            lora_dropout=0.05,  # Lowered dropout
            target_modules=["q_proj", "v_proj", "k_proj"]  # Improved selection
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.to(self.device)

        # Evaluation metric
        self.metric = evaluate.load("accuracy")

    def load_dataset(self, dataset_path="datasets/travel_training_data.json"):
        """Load and preprocess dataset."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        train_size = int(0.9 * len(data))  # 90% train, 10% validation
        train_data = data[:train_size]
        eval_data = data[train_size:]

        def format_example(example):
            return {
                "text": f"User: {example['instruction']}\nBot: {example['response']}"
            }

        train_dataset = Dataset.from_list([format_example(item) for item in train_data])
        eval_dataset = Dataset.from_list([format_example(item) for item in eval_data])

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,  # Prevents excessive length
                padding="max_length"
            )

        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        return train_dataset, eval_dataset

    def compute_metrics(self, eval_pred):
        """Custom metric evaluation for better training performance."""
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def train(self, train_dataset, eval_dataset=None, output_dir="./travel_llm_model"):
        """Train the model with improved settings."""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,  # Increased epochs for better learning
            per_device_train_batch_size=4,  # Increased batch size
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=50,
            eval_steps=250,
            save_steps=500,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False,
            fp16=True if self.device == "cuda" else False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )

        trainer.train()

        # Save trained model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return trainer

    def optimize_for_mobile(self, output_dir="./travel_llm_mobile"):
        """Optimize and save model for mobile deployment."""
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        # Prune model for size reduction
        from torch.nn.utils import prune
        for name, module in quantized_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.3)  # Prune 30% weights

        os.makedirs(output_dir, exist_ok=True)
        
        # Save quantized model properly
        torch.save(quantized_model.state_dict(), os.path.join(output_dir, "quantized_model.pt"))
        self.tokenizer.save_pretrained(output_dir)

        return quantized_model

# Main Execution
if __name__ == "__main__":
    trainer = TravelLLMTrainer()
    train_dataset, eval_dataset = trainer.load_dataset()
    trainer.train(train_dataset, eval_dataset)
    trainer.optimize_for_mobile()

    print("Training complete. Model saved to ./travel_llm_model")
    print("Optimized model saved to ./travel_llm_mobile")
