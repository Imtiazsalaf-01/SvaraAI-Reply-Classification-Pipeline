import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import matplotlib.pyplot as plt

def load_data():
    """Load preprocessed data"""
    try:
        train_df = pd.read_csv("data/train.csv")
        val_df = pd.read_csv("data/val.csv")
        test_df = pd.read_csv("data/test.csv")
        print("Loaded preprocessed data splits")
    except FileNotFoundError:
        print("Running data preprocessing first...")
        os.system("python src/data_preprocessing.py")
        train_df = pd.read_csv("data/train.csv")
        val_df = pd.read_csv("data/val.csv")
        test_df = pd.read_csv("data/test.csv")
    
    return train_df, val_df, test_df

def prepare_datasets(train_df, val_df, test_df, tokenizer, max_length=128):
    """Prepare datasets for training"""
    
    # Create label mappings
    labels = sorted(train_df['label'].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    print(f"Label mappings: {label2id}")
    
    # Add label IDs to dataframes
    train_df['label_id'] = train_df['label'].map(label2id)
    val_df['label_id'] = val_df['label'].map(label2id)
    test_df['label_id'] = test_df['label'].map(label2id)
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df[['reply', 'label_id']].rename(columns={'label_id': 'labels'}))
    val_dataset = Dataset.from_pandas(val_df[['reply', 'label_id']].rename(columns={'label_id': 'labels'}))
    test_dataset = Dataset.from_pandas(test_df[['reply', 'label_id']].rename(columns={'label_id': 'labels'}))
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples['reply'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return train_dataset, val_dataset, test_dataset, label2id, id2label

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def plot_training_history(trainer):
    """Plot training history"""
    log_history = trainer.state.log_history
    
    train_loss = []
    eval_loss = []
    eval_f1 = []
    
    for log in log_history:
        if 'loss' in log:
            train_loss.append(log['loss'])
        if 'eval_loss' in log:
            eval_loss.append(log['eval_loss'])
        if 'eval_f1_macro' in log:
            eval_f1.append(log['eval_f1_macro'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_loss, label='Training Loss', marker='o')
    ax1.plot(eval_loss, label='Validation Loss', marker='s')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot F1 score
    ax2.plot(eval_f1, label='Validation F1 (Macro)', marker='o', color='green')
    ax2.set_title('Validation F1 Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('transformer_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(trainer, test_dataset, id2label):
    """Evaluate the trained model"""
    print("\nEvaluating on test set...")
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    # Convert to label names
    y_pred_labels = [id2label[pred] for pred in y_pred]
    y_true_labels = [id2label[true] for true in y_true]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_true_labels, y_pred_labels)}")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': y_pred_labels,
        'true_labels': y_true_labels
    }

def main():
    print("Training Transformer Model (DistilBERT)")
    print("=" * 50)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset, label2id, id2label = prepare_datasets(
        train_df, val_df, test_df, tokenizer
    )
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
        dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
        report_to=None  # Disable wandb logging
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train model
    print("\nStarting training...")
    trainer.train()
    
    # Plot training history
    plot_training_history(trainer)
    
    # Evaluate on test set
    test_results = evaluate_model(trainer, test_dataset, id2label)
    
    # Save model and tokenizer
    os.makedirs("models/transformer", exist_ok=True)
    trainer.save_model("models/transformer")
    tokenizer.save_pretrained("models/transformer")
    
    print("\nModel saved to 'models/transformer'")
    
    # Save results
    results = {
        'model_type': 'transformer',
        'model_name': model_name,
        'test_accuracy': test_results['accuracy'],
        'test_f1_macro': test_results['f1_macro'],
        'test_f1_weighted': test_results['f1_weighted']
    }
    
    pd.DataFrame([results]).to_csv("models/transformer_results.csv", index=False)
    
    print(f"\nFinal Test Results:")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Macro F1: {test_results['f1_macro']:.4f}")

if __name__ == "__main__":
    main()