#!/usr/bin/env python3
"""
SvaraAI ML Engineer Internship Assignment
Reply Classification Pipeline - Part A

This script implements the complete ML/NLP pipeline for classifying email replies
into positive, negative, and neutral categories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os

# For transformer model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

def load_and_preprocess_data(file_path="data/reply_classification_dataset.csv"):
    """
    Load and preprocess the dataset
    """
    print("Loading and preprocessing dataset...")
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Check for missing values
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Drop rows with missing values
    df = df.dropna(subset=['reply', 'label']).copy()
    
    # Normalize labels
    df['label'] = df['label'].str.lower().str.strip()
    
    # Map label variations to standard labels
    label_mapping = {
        'positive': 'positive', 'pos': 'positive', 'positive.': 'positive',
        'negative': 'negative', 'neg': 'negative', 'negative.': 'negative',
        'neutral': 'neutral', 'neutal': 'neutral', 'neautral': 'neutral'
    }
    df['label'] = df['label'].map(label_mapping).fillna(df['label'])
    
    # Filter to only expected classes
    valid_labels = ['positive', 'negative', 'neutral']
    df = df[df['label'].isin(valid_labels)].reset_index(drop=True)
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df

def create_data_splits(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create stratified train/validation/test splits
    """
    print("\nCreating data splits...")
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio, stratify=train_val_df['label'], random_state=random_state
    )
    
    print(f"Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def train_baseline_model(train_df, val_df):
    """
    Train baseline model using TF-IDF + Logistic Regression
    """
    print("\nTraining baseline model (TF-IDF + Logistic Regression)...")
    
    # Prepare data
    X_train = train_df['reply'].values
    y_train = train_df['label'].values
    X_val = val_df['reply'].values
    y_val = val_df['label'].values
    
    # Create pipeline
    baseline_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            stop_words='english',
            lowercase=True
        )),
        ('classifier', LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            C=1.0,
            random_state=42
        ))
    ])
    
    # Train model
    baseline_pipeline.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_predictions = baseline_pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions, average='macro')
    
    print(f"Baseline Validation Results:")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Macro F1: {val_f1:.4f}")
    
    return baseline_pipeline

def train_transformer_model(train_df, val_df):
    """
    Train transformer model using DistilBERT
    """
    print("\nTraining transformer model (DistilBERT)...")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create label mappings
    labels = sorted(train_df['label'].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    # Add label IDs
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df['label_id'] = train_df['label'].map(label2id)
    val_df['label_id'] = val_df['label'].map(label2id)
    
    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples['reply'], truncation=True, padding='max_length', max_length=128)
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df[['reply', 'label_id']].rename(columns={'label_id': 'labels'}))
    val_dataset = Dataset.from_pandas(val_df[['reply', 'label_id']].rename(columns={'label_id': 'labels'}))
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to=None
    )
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='macro')
        }
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Train model
    trainer.train()
    
    # Save model
    os.makedirs("models/transformer", exist_ok=True)
    trainer.save_model("models/transformer")
    tokenizer.save_pretrained("models/transformer")
    
    return trainer, tokenizer, id2label

def evaluate_models(baseline_model, test_df, transformer_model=None, tokenizer=None, id2label=None):
    """
    Evaluate both models on test set
    """
    print("\nEvaluating models on test set...")
    
    X_test = test_df['reply'].values
    y_test = test_df['label'].values
    
    # Evaluate baseline model
    baseline_predictions = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    baseline_f1 = f1_score(y_test, baseline_predictions, average='macro')
    
    print(f"\nBaseline Model Results:")
    print(f"Accuracy: {baseline_accuracy:.4f}")
    print(f"Macro F1: {baseline_f1:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, baseline_predictions)}")
    
    results = {
        'baseline': {
            'accuracy': baseline_accuracy,
            'f1_macro': baseline_f1,
            'predictions': baseline_predictions
        }
    }
    
    # Evaluate transformer model if available
    if transformer_model and tokenizer and id2label:
        print(f"\nEvaluating transformer model...")
        
        # Get predictions from transformer
        transformer_predictions = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transformer_model.model.to(device)
        transformer_model.model.eval()
        
        for text in X_test:
            inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = transformer_model.model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
                transformer_predictions.append(id2label[prediction])
        
        transformer_accuracy = accuracy_score(y_test, transformer_predictions)
        transformer_f1 = f1_score(y_test, transformer_predictions, average='macro')
        
        print(f"Transformer Model Results:")
        print(f"Accuracy: {transformer_accuracy:.4f}")
        print(f"Macro F1: {transformer_f1:.4f}")
        print(f"Classification Report:\n{classification_report(y_test, transformer_predictions)}")
        
        results['transformer'] = {
            'accuracy': transformer_accuracy,
            'f1_macro': transformer_f1,
            'predictions': transformer_predictions
        }
    
    return results

def create_visualizations(test_df, results):
    """
    Create visualizations for model comparison
    """
    print("\nCreating visualizations...")
    
    y_test = test_df['label'].values
    
    # Create confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Baseline confusion matrix
    cm_baseline = confusion_matrix(y_test, results['baseline']['predictions'])
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Baseline Model - Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Model comparison
    if 'transformer' in results:
        cm_transformer = confusion_matrix(y_test, results['transformer']['predictions'])
        sns.heatmap(cm_transformer, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title('Transformer Model - Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
    else:
        # Show performance comparison
        models = ['Baseline']
        accuracies = [results['baseline']['accuracy']]
        f1_scores = [results['baseline']['f1_macro']]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[1].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        axes[1].bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Model Performance Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models)
        axes[1].legend()
        axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the complete pipeline
    """
    print("🚀 SvaraAI Reply Classification Pipeline")
    print("=" * 50)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Create data splits
    train_df, val_df, test_df = create_data_splits(df)
    
    # Save data splits
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    
    # Train baseline model
    baseline_model = train_baseline_model(train_df, val_df)
    
    # Save baseline model
    joblib.dump(baseline_model, "models/baseline_model.joblib")
    print("Baseline model saved to models/baseline_model.joblib")
    
    # Train transformer model (optional - can be commented out for faster execution)
    try:
        transformer_model, tokenizer, id2label = train_transformer_model(train_df, val_df)
        print("Transformer model saved to models/transformer/")
    except Exception as e:
        print(f"Transformer training failed (this is optional): {e}")
        transformer_model, tokenizer, id2label = None, None, None
    
    # Evaluate models
    results = evaluate_models(baseline_model, test_df, transformer_model, tokenizer, id2label)
    
    # Create visualizations
    create_visualizations(test_df, results)
    
    # Model comparison and recommendation
    print("\n" + "=" * 50)
    print("MODEL COMPARISON AND RECOMMENDATION")
    print("=" * 50)
    
    baseline_acc = results['baseline']['accuracy']
    baseline_f1 = results['baseline']['f1_macro']
    
    print(f"Baseline Model Performance:")
    print(f"  - Accuracy: {baseline_acc:.4f}")
    print(f"  - F1 Score: {baseline_f1:.4f}")
    print(f"  - Model Size: ~34KB")
    print(f"  - Inference Speed: Very Fast (~1-5ms)")
    
    if 'transformer' in results:
        transformer_acc = results['transformer']['accuracy']
        transformer_f1 = results['transformer']['f1_macro']
        
        print(f"\nTransformer Model Performance:")
        print(f"  - Accuracy: {transformer_acc:.4f}")
        print(f"  - F1 Score: {transformer_f1:.4f}")
        print(f"  - Model Size: ~250MB")
        print(f"  - Inference Speed: Slower (~20-50ms)")
        
        # Recommendation
        if transformer_acc > baseline_acc + 0.02:  # 2% improvement threshold
            print(f"\n🎯 RECOMMENDATION: Use Transformer Model")
            print(f"   The transformer model shows significant improvement ({transformer_acc:.4f} vs {baseline_acc:.4f})")
        else:
            print(f"\n🎯 RECOMMENDATION: Use Baseline Model")
            print(f"   The baseline model offers similar performance with much better efficiency")
    else:
        print(f"\n🎯 RECOMMENDATION: Use Baseline Model")
        print(f"   Excellent performance ({baseline_acc:.4f} accuracy) with high efficiency")
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"   - Models saved in models/ directory")
    print(f"   - Data splits saved in data/ directory")
    print(f"   - Ready for deployment with app.py")

if __name__ == "__main__":
    main()