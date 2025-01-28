from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Step 1: Load the DeepSeek-R1 model and tokenizer
model_name = "DeepSeek-R1-bf16"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Load and preprocess the dataset
# Ensure your data is formatted as JSON: [{"instruction": "...", "input": "...", "output": "..."}]
dataset = load_dataset("json", data_files="skyline_data.json")

# Preprocess the dataset: Tokenize instructions and outputs
def preprocess_function(examples):
    # Combine instruction and input, using the output as the target text
    inputs = [f"{example['instruction']} {example['input']}" for example in examples]
    targets = [example["output"] for example in examples]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Step 3: Define training arguments
training_args = TrainingArguments(
    output_dir="./trained_deepseek",       # Directory to save the model
    evaluation_strategy="steps",          # Evaluate during training
    eval_steps=500,                       # Evaluate every 500 steps
    save_steps=500,                       # Save checkpoint every 500 steps
    save_total_limit=2,                   # Keep only the 2 latest checkpoints
    per_device_train_batch_size=2,        # Adjust based on GPU memory
    per_device_eval_batch_size=2,
    num_train_epochs=3,                   # Number of epochs
    logging_dir="./logs",                 # Log directory
    logging_steps=10,                     # Log every 10 steps
    bf16=True,                            # Enable Brain Float 16 for efficiency
    report_to="none",                     # No reporting to external services
    fp16_full_eval=False,                 # No mixed precision during evaluation
)

# Step 4: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# Step 5: Train the model
trainer.train()

# Step 6: Save the fine-tuned model and tokenizer
model.save_pretrained("./trained_deepseek")
tokenizer.save_pretrained("./trained_deepseek")
