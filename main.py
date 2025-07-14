from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import torch

with open("data.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

dataset = Dataset.from_dict({"text": lines})

def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("🔧 Starting training...")
trainer.train()
print("✅ Training complete!")

model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")
print("💾 Model saved to ./gpt2-finetuned")

prompt = "In a world of silence"
inputs = tokenizer(prompt, return_tensors="pt")

if torch.cuda.is_available():
    model = model.to("cuda")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

print("\n🎤 Generating sample text...\n")
output = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("📝 Output:\n", generated_text)
