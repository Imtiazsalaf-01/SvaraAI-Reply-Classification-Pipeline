import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load preprocessed data or create splits if not available"""
    try:
        train_df = pd.read_csv("data/train.csv")
        val_df = pd.read_csv("data/val.csv") 
        test_df = pd.read_csv("data/test.csv")
        print("Loaded preprocessed data splits")
    except FileNotFoundError:
        print("Preprocessed splits not found, creating from original data...")
        df = pd.read_csv("data/reply_classification_dataset.csv")
        df = df.dropna(subset=['reply', 'label'])
        df['label'] = df['label'].str.lower().str.strip()
        df = df[df['label'].isin(['positive', 'negative', 'neutral'])]
        
        train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.125, stratify=train_val_df['label'], random_state=42)
    
    return train_df, val_df, test_df

def create_baseline_model():
    """Create TF-IDF + Logistic Regression pipeline"""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),  # Use unigrams and bigrams
            max_features=10000,   # Limit vocabulary size
            stop_words='english', # Remove common English stop words
            lowercase=True,       # Convert to lowercase
            strip_accents='ascii' # Remove accents
        )),
        ('classifier', LogisticRegression(
            max_iter=2000,
            class_weight='balanced',  # Handle class imbalance
            C=1.0,                   # Regularization strength
            random_state=42
        ))
    ])
    return pipeline

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model and print metrics"""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average='macro')
    f1_weighted = f1_score(y_test, predictions, average='weighted')
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, predictions)}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=300)
    plt.show()
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': predictions,
        'probabilities': probabilities
    }

def analyze_feature_importance(pipeline, top_n=20):
    """Analyze most important features for each class"""
    vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['classifier']
    
    feature_names = vectorizer.get_feature_names_out()
    
    plt.figure(figsize=(15, 10))
    
    for i, class_name in enumerate(classifier.classes_):
        coefficients = classifier.coef_[i]
        
        # Get top positive and negative features
        top_positive_idx = np.argsort(coefficients)[-top_n:]
        top_negative_idx = np.argsort(coefficients)[:top_n]
        
        plt.subplot(2, 2, i+1)
        
        # Plot top features
        top_features = np.concatenate([top_negative_idx, top_positive_idx])
        top_coeffs = coefficients[top_features]
        top_names = [feature_names[idx] for idx in top_features]
        
        colors = ['red' if coef < 0 else 'green' for coef in top_coeffs]
        
        plt.barh(range(len(top_coeffs)), top_coeffs, color=colors, alpha=0.7)
        plt.yticks(range(len(top_coeffs)), top_names)
        plt.title(f'Top Features for {class_name.title()}')
        plt.xlabel('Coefficient Value')
        
    plt.tight_layout()
    plt.savefig('baseline_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Training Baseline Model (TF-IDF + Logistic Regression)")
    print("=" * 60)
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Prepare data
    X_train = train_df['reply'].values
    y_train = train_df['label'].values
    X_val = val_df['reply'].values
    y_val = val_df['label'].values
    X_test = test_df['reply'].values
    y_test = test_df['label'].values
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create and train model
    print("\nTraining baseline model...")
    pipeline = create_baseline_model()
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_results = evaluate_model(pipeline, X_val, y_val, "Baseline (Validation)")
    
    # Evaluate on test set
    test_results = evaluate_model(pipeline, X_test, y_test, "Baseline (Test)")
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    analyze_feature_importance(pipeline)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/baseline_model.joblib")
    print("\nModel saved as 'models/baseline_model.joblib'")
    
    # Save results
    results = {
        'model_type': 'baseline',
        'validation_accuracy': val_results['accuracy'],
        'validation_f1_macro': val_results['f1_macro'],
        'test_accuracy': test_results['accuracy'],
        'test_f1_macro': test_results['f1_macro']
    }
    
    pd.DataFrame([results]).to_csv("models/baseline_results.csv", index=False)
    
    print(f"\nFinal Test Results:")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Macro F1: {test_results['f1_macro']:.4f}")

if __name__ == "__main__":
    main()