import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(file_path):
    """Load and clean the dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Drop rows with missing values
    df = df.dropna(subset=['reply', 'label']).copy()
    print(f"After removing missing values: {df.shape}")
    
    # Normalize labels
    df['label'] = df['label'].str.lower().str.strip()
    
    # Map label variations to standard labels
    label_mapping = {
        'positive': 'positive',
        'pos': 'positive',
        'positive.': 'positive',
        'negative': 'negative', 
        'neg': 'negative',
        'negative.': 'negative',
        'neutral': 'neutral',
        'neutal': 'neutral',
        'neautral': 'neutral'
    }
    
    df['label'] = df['label'].map(label_mapping).fillna(df['label'])
    
    # Filter to only expected classes
    valid_labels = ['positive', 'negative', 'neutral']
    df = df[df['label'].isin(valid_labels)].reset_index(drop=True)
    
    print(f"Final dataset shape: {df.shape}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    
    return df

def create_train_test_split(df, test_size=0.2, val_size=0.1, random_state=42):
    """Create stratified train/validation/test splits"""
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['label'], 
        random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)  # Adjust val_size for remaining data
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        stratify=train_val_df['label'],
        random_state=random_state
    )
    
    print(f"\nData splits:")
    print(f"Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Check label distribution in each split
    print(f"\nTrain label distribution:\n{train_df['label'].value_counts()}")
    print(f"\nValidation label distribution:\n{val_df['label'].value_counts()}")
    print(f"\nTest label distribution:\n{test_df['label'].value_counts()}")
    
    return train_df, val_df, test_df

def visualize_data(df):
    """Create visualizations of the dataset"""
    plt.figure(figsize=(12, 8))
    
    # Label distribution
    plt.subplot(2, 2, 1)
    df['label'].value_counts().plot(kind='bar')
    plt.title('Label Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Reply length distribution
    plt.subplot(2, 2, 2)
    df['reply_length'] = df['reply'].str.len()
    plt.hist(df['reply_length'], bins=30, alpha=0.7)
    plt.title('Reply Length Distribution')
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    
    # Reply length by sentiment
    plt.subplot(2, 2, 3)
    for label in df['label'].unique():
        subset = df[df['label'] == label]['reply_length']
        plt.hist(subset, alpha=0.5, label=label, bins=20)
    plt.title('Reply Length by Sentiment')
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Word count distribution
    plt.subplot(2, 2, 4)
    df['word_count'] = df['reply'].str.split().str.len()
    df.boxplot(column='word_count', by='label', ax=plt.gca())
    plt.title('Word Count by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Word Count')
    
    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_clean_data("data/reply_classification_dataset.csv")
    
    # Create visualizations
    df = visualize_data(df)
    
    # Create train/test splits
    train_df, val_df, test_df = create_train_test_split(df)
    
    # Save processed data
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    
    print("\nData preprocessing completed!")
    print("Saved files: train.csv, val.csv, test.csv")