import os
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
import json

# Configuration
class Config:
    MODEL_NAME = "google/flan-t5-base"
    DATASET_PATH = "/kaggle/input/medical-chat-summarization/medical_chat.csv"
    OUTPUT_DIR = "./medical-chat-summarizer"
    
    # LoRA config
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    
    # Training config
    MAX_SOURCE_LENGTH = 512
    MAX_TARGET_LENGTH = 256
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500

def load_data(file_path):
    """Load and prepare dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} samples")
        
        # Standardize column names
        if 'soap' in df.columns:
            df = df.rename(columns={'soap': 'summary'})
        elif 'SOAP' in df.columns:
            df = df.rename(columns={'SOAP': 'summary'})
        
        # Remove missing values
        df = df.dropna(subset=['dialogue', 'summary'])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_datasets(df, test_size=0.1, val_size=0.1):
    """Split data into train/val/test"""
    # Split
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size / (1 - test_size), 
        random_state=42
    )
    
    # Convert to Dataset
    train_dataset = Dataset.from_pandas(train_df[['dialogue', 'summary']].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[['dialogue', 'summary']].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df[['dialogue', 'summary']].reset_index(drop=True))
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

def preprocess_function(examples, tokenizer, max_source_length, max_target_length):
    """Tokenize examples"""
    # Add instruction
    inputs = [
        f"Summarize this medical conversation in SOAP format:\n\n{dialogue}\n\nSummary:"
        for dialogue in examples['dialogue']
    ]
    
    # Tokenize
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        examples['summary'],
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    """Main training function"""
    config = Config()
    
    print("=" * 80)
    print("Medical Chat Summarization - Training")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading dataset...")
    df = load_data(config.DATASET_PATH)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Prepare datasets
    print("\n2. Preparing datasets...")
    datasets = prepare_datasets(df)
    print(f"Train: {len(datasets['train'])}, Val: {len(datasets['validation'])}, Test: {len(datasets['test'])}")
    
    # Load tokenizer
    print("\n3. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Tokenize datasets
    print("\n4. Tokenizing datasets...")
    tokenized_datasets = datasets.map(
        lambda x: preprocess_function(x, tokenizer, config.MAX_SOURCE_LENGTH, config.MAX_TARGET_LENGTH),
        batched=True,
        remove_columns=datasets["train"].column_names
    )
    
    # Load model
    print("\n5. Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.MODEL_NAME,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Apply LoRA
    print("\n6. Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["q", "v"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training arguments
    print("\n7. Setting up training...")
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train
    print("\n8. Starting training...")
    print("=" * 80)
    trainer.train()
    
    # Save model
    print("\n9. Saving model...")
    trainer.save_model(f"{config.OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/final_model")
    
    # Save config
    config_dict = {
        'model_name': config.MODEL_NAME,
        'max_source_length': config.MAX_SOURCE_LENGTH,
        'max_target_length': config.MAX_TARGET_LENGTH,
        'lora_r': config.LORA_R,
        'lora_alpha': config.LORA_ALPHA
    }
    
    with open(f"{config.OUTPUT_DIR}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Model saved to: {config.OUTPUT_DIR}/final_model")
    print("=" * 80)

if __name__ == "__main__":
    main()
