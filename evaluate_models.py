import pandas as pd
import numpy as np
import joblib
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

def load_test_data():
    """Load test data"""
    try:
        test_df = pd.read_csv("data/test.csv")
    except FileNotFoundError:
        print("Test data not found. Please run data preprocessing first.")
        return None
    return test_df

def evaluate_baseline_model(test_df):
    """Evaluate baseline model"""
    print("Evaluating Baseline Model...")
    
    try:
        # Load model
        baseline_model = joblib.load("models/baseline_model.joblib")
        
        X_test = test_df['reply'].values
        y_test = test_df['label'].values
        
        # Time prediction
        start_time = time.time()
        predictions = baseline_model.predict(X_test)
        probabilities = baseline_model.predict_proba(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average='macro')
        f1_weighted = f1_score(y_test, predictions, average='weighted')
        
        results = {
            'model': 'Baseline (TF-IDF + LogReg)',
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'prediction_time': prediction_time,
            'avg_time_per_sample': prediction_time / len(X_test),
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        print(f"Baseline Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Prediction time: {prediction_time:.4f}s")
        print(f"Avg time per sample: {prediction_time/len(X_test)*1000:.2f}ms")
        
        return results
        
    except FileNotFoundError:
        print("Baseline model not found. Please train it first.")
        return None

def evaluate_transformer_model(test_df):
    """Evaluate transformer model"""
    print("\nEvaluating Transformer Model...")
    
    # Check if transformer model exists
    if not os.path.exists("models/transformer"):
        print("Transformer model not found. Please train it first with: python src/train_transformer.py")
        return None
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("models/transformer")
        model = AutoModelForSequenceClassification.from_pretrained("models/transformer")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        X_test = test_df['reply'].values
        y_test = test_df['label'].values
        
        # Get label mappings
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        
        # Time prediction
        start_time = time.time()
        predictions = []
        probabilities = []
        
        batch_size = 32
        for i in range(0, len(X_test), batch_size):
            batch_texts = X_test[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                list(batch_texts),
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                batch_probs = probs.cpu().numpy()
                
                predictions.extend(batch_preds)
                probabilities.extend(batch_probs)
        
        prediction_time = time.time() - start_time
        
        # Convert predictions to labels
        pred_labels = [id2label[pred] for pred in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, pred_labels)
        f1_macro = f1_score(y_test, pred_labels, average='macro')
        f1_weighted = f1_score(y_test, pred_labels, average='weighted')
        
        results = {
            'model': 'Transformer (DistilBERT)',
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'prediction_time': prediction_time,
            'avg_time_per_sample': prediction_time / len(X_test),
            'predictions': pred_labels,
            'probabilities': np.array(probabilities)
        }
        
        print(f"Transformer Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Prediction time: {prediction_time:.4f}s")
        print(f"Avg time per sample: {prediction_time/len(X_test)*1000:.2f}ms")
        
        return results
        
    except FileNotFoundError:
        print("Transformer model not found. Please train it first.")
        return None

def compare_models(baseline_results, transformer_results, test_df):
    """Compare both models and create visualizations"""
    
    if baseline_results is None or transformer_results is None:
        print("Cannot compare models - one or both models not available")
        return
    
    # Create comparison dataframe
    comparison_data = {
        'Model': [baseline_results['model'], transformer_results['model']],
        'Accuracy': [baseline_results['accuracy'], transformer_results['accuracy']],
        'Macro F1': [baseline_results['f1_macro'], transformer_results['f1_macro']],
        'Weighted F1': [baseline_results['f1_weighted'], transformer_results['f1_weighted']],
        'Avg Time (ms)': [baseline_results['avg_time_per_sample']*1000, 
                         transformer_results['avg_time_per_sample']*1000]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(f"\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv("models/model_comparison.csv", index=False)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    axes[0, 0].bar(comparison_df['Model'], comparison_df['Accuracy'], 
                   color=['skyblue', 'lightcoral'])
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(comparison_df['Accuracy']):
        axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # F1 Score comparison
    x = np.arange(len(comparison_df))
    width = 0.35
    axes[0, 1].bar(x - width/2, comparison_df['Macro F1'], width, 
                   label='Macro F1', color='lightgreen')
    axes[0, 1].bar(x + width/2, comparison_df['Weighted F1'], width, 
                   label='Weighted F1', color='orange')
    axes[0, 1].set_title('F1 Score Comparison')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(comparison_df['Model'])
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)
    
    # Inference time comparison
    axes[1, 0].bar(comparison_df['Model'], comparison_df['Avg Time (ms)'], 
                   color=['gold', 'lightpink'])
    axes[1, 0].set_title('Average Inference Time per Sample')
    axes[1, 0].set_ylabel('Time (ms)')
    for i, v in enumerate(comparison_df['Avg Time (ms)']):
        axes[1, 0].text(i, v + 0.1, f'{v:.2f}ms', ha='center')
    
    # Confusion matrices
    y_true = test_df['label'].values
    
    # Baseline confusion matrix
    cm_baseline = confusion_matrix(y_true, baseline_results['predictions'])
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', 
                ax=axes[1, 1], cbar=False)
    axes[1, 1].set_title('Baseline - Confusion Matrix')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed classification reports
    print(f"\nBaseline Classification Report:")
    print(classification_report(y_true, baseline_results['predictions']))
    
    print(f"\nTransformer Classification Report:")
    print(classification_report(y_true, transformer_results['predictions']))
    
    # Performance summary
    print(f"\nPerformance Summary:")
    print(f"Best Accuracy: {comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']}")
    print(f"Best Macro F1: {comparison_df.loc[comparison_df['Macro F1'].idxmax(), 'Model']}")
    print(f"Fastest Inference: {comparison_df.loc[comparison_df['Avg Time (ms)'].idxmin(), 'Model']}")
    
    return comparison_df

def analyze_errors(baseline_results, transformer_results, test_df):
    """Analyze prediction errors"""
    if baseline_results is None or transformer_results is None:
        return
    
    y_true = test_df['label'].values
    baseline_preds = baseline_results['predictions']
    transformer_preds = transformer_results['predictions']
    
    # Find samples where models disagree
    disagreement_mask = baseline_preds != transformer_preds
    disagreement_samples = test_df[disagreement_mask].copy()
    disagreement_samples['baseline_pred'] = baseline_preds[disagreement_mask]
    disagreement_samples['transformer_pred'] = transformer_preds[disagreement_mask]
    disagreement_samples['true_label'] = y_true[disagreement_mask]
    
    print(f"\nModel Disagreement Analysis:")
    print(f"Samples where models disagree: {len(disagreement_samples)} ({len(disagreement_samples)/len(test_df)*100:.1f}%)")
    
    if len(disagreement_samples) > 0:
        print(f"\nSample disagreements:")
        for i, row in disagreement_samples.head(5).iterrows():
            print(f"Text: '{row['reply'][:100]}...'")
            print(f"True: {row['true_label']}, Baseline: {row['baseline_pred']}, Transformer: {row['transformer_pred']}")
            print("-" * 50)
    
    return disagreement_samples

def main():
    print("Model Evaluation and Comparison")
    print("=" * 40)
    
    # Load test data
    test_df = load_test_data()
    if test_df is None:
        return
    
    print(f"Test samples: {len(test_df)}")
    
    # Evaluate baseline model
    baseline_results = evaluate_baseline_model(test_df)
    
    # Try to evaluate transformer model
    transformer_results = evaluate_transformer_model(test_df)
    
    # Compare models if both are available
    if baseline_results and transformer_results:
        comparison_df = compare_models(baseline_results, transformer_results, test_df)
        
        # Analyze errors
        disagreement_samples = analyze_errors(baseline_results, transformer_results, test_df)
        
        # Save disagreement analysis
        if len(disagreement_samples) > 0:
            disagreement_samples.to_csv("models/model_disagreements.csv", index=False)
    elif baseline_results:
        print("\nOnly baseline model available for evaluation.")
        print("To train transformer model, run: python src/train_transformer.py")
        
        # Create simple baseline summary
        summary_data = {
            'Model': ['Baseline (TF-IDF + LogReg)'],
            'Accuracy': [baseline_results['accuracy']],
            'Macro F1': [baseline_results['f1_macro']],
            'Weighted F1': [baseline_results['f1_weighted']],
            'Avg Time (ms)': [baseline_results['avg_time_per_sample']*1000]
        }
        
        summary_df = pd.DataFrame(summary_data)
        print(f"\nBaseline Model Summary:")
        print(summary_df.to_string(index=False))
        summary_df.to_csv("models/baseline_summary.csv", index=False)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()