"""
Demo Test Script for Active Learning Agent
Simulates the competition environment to verify the agent works correctly
"""

import numpy as np
import pandas as pd
from agent import run_agent
import time


def generate_synthetic_data(n_samples=10000, n_features=20, fraud_rate=0.08):
    """
    Generate synthetic fraud detection dataset
    
    Args:
        n_samples: Number of samples (default 10,000)
        n_features: Number of features (default 20)
        fraud_rate: Percentage of fraudulent samples (default 8%)
    
    Returns:
        df: Feature DataFrame
        true_labels: True labels array
    """
    np.random.seed(42)
    
    # Calculate number of fraud cases
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud
    
    print(f"Generating synthetic dataset:")
    print(f"  Total samples: {n_samples}")
    print(f"  Legitimate: {n_legit} ({(1-fraud_rate)*100:.1f}%)")
    print(f"  Fraudulent: {n_fraud} ({fraud_rate*100:.1f}%)")
    print(f"  Features: {n_features}")
    
    # Generate legitimate profiles (normal distribution)
    X_legit = np.random.randn(n_legit, n_features)
    
    # Generate fraudulent profiles (anomalous - different distribution)
    # Fraud profiles have higher variance and different means
    X_fraud = np.random.randn(n_fraud, n_features) * 2.5 + np.random.uniform(-3, 3, n_features)
    
    # Combine and shuffle
    X = np.vstack([X_legit, X_fraud])
    y = np.hstack([np.zeros(n_legit), np.ones(n_fraud)])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Create DataFrame with feature names
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    return df, y.astype(int)


def create_oracle(true_labels):
    """
    Create an oracle function that returns true labels for queried indices
    
    Args:
        true_labels: Array of all true labels
    
    Returns:
        oracle_fn: Function that takes indices and returns labels
    """
    query_count = [0]  # Mutable counter
    
    def oracle_fn(indices):
        """Oracle function that returns labels for requested indices"""
        query_count[0] += len(indices)
        print(f"\n📋 Oracle queried for {len(indices)} samples")
        print(f"   Total queries so far: {query_count[0]}")
        return [true_labels[i] for i in indices]
    
    oracle_fn.query_count = query_count
    return oracle_fn


def evaluate_predictions(y_true, y_pred):
    """
    Calculate evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        dict: Evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, confusion_matrix, classification_report
    )
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }


def main():
    """Main test function"""
    print("=" * 70)
    print("ACTIVE LEARNING AGENT - DEMO TEST")
    print("=" * 70)
    print()
    
    # Step 1: Load real dataset
    print("Step 1: Loading Real Dataset")
    print("-" * 70)
    print("Loading dataset from: dataset.csv")
    df = pd.read_csv('dataset.csv')
    print(f"✓ Dataset loaded: {df.shape}")
    print(f"   Features: {list(df.columns[:5])}... (showing first 5)")
    
    # Generate synthetic labels for testing (since we don't have real labels.npy yet)
    # In actual competition, labels would come from labels.npy
    print("\nGenerating synthetic labels for testing...")
    np.random.seed(42)
    
    # Create realistic fraud labels based on suspicious patterns
    fraud_score = (
        df['applications_7d'].values / (df['applications_7d'].max() + 1) * 0.2 +
        df['ip_risk_score'].values * 0.15 +
        df['email_risk_score'].values * 0.15 +
        df['institution_risk_score'].values * 0.1 +
        df['company_risk_score'].values * 0.1 +
        df['copy_paste_ratio'].values * 0.1 +
        df['login_velocity_24h'].values / (df['login_velocity_24h'].max() + 1) * 0.1 +
        df['failed_logins_24h'].values / (df['failed_logins_24h'].max() + 1) * 0.1
    )
    fraud_score += np.random.rand(len(df)) * 0.2
    
    # Top 8% are labeled as fraud
    fraud_threshold = np.percentile(fraud_score, 92)
    true_labels = (fraud_score > fraud_threshold).astype(int)
    
    print(f"✓ Labels generated: {len(true_labels)} total, {true_labels.sum()} fraudulent ({true_labels.sum()/len(true_labels)*100:.1f}%)")
    print()
    
    # Step 2: Create oracle
    print("Step 2: Creating Oracle Function")
    print("-" * 70)
    oracle_fn = create_oracle(true_labels)
    budget = 100
    print(f"✓ Oracle ready (Budget: {budget} queries)")
    print()
    
    # Step 3: Run agent
    print("Step 3: Running Active Learning Agent")
    print("-" * 70)
    print("Agent Strategy: Anomaly-First Learning")
    print("  1. Isolation Forest identifies 100 most suspicious profiles")
    print("  2. Oracle labels these 100 samples")
    print("  3. Random Forest trains on labeled data")
    print("  4. Predictions made for all 10,000 samples")
    print()
    
    start_time = time.time()
    
    try:
        predictions = run_agent(df, oracle_fn, budget)
        
        runtime = time.time() - start_time
        
        print(f"\n✓ Agent completed successfully!")
        print(f"   Runtime: {runtime:.2f} seconds")
        print(f"   Queries used: {oracle_fn.query_count[0]}/{budget}")
        print()
        
        # Step 4: Evaluate results
        print("Step 4: Evaluating Predictions")
        print("-" * 70)
        
        metrics = evaluate_predictions(true_labels, predictions)
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1 Score:  {metrics['f1_score']:.4f}")
        print()
        
        print(f"📈 CONFUSION MATRIX:")
        print(f"                  Predicted")
        print(f"                Legit  Fraud")
        print(f"   Actual Legit  {metrics['confusion_matrix'][0][0]:5d}  {metrics['confusion_matrix'][0][1]:5d}")
        print(f"          Fraud  {metrics['confusion_matrix'][1][0]:5d}  {metrics['confusion_matrix'][1][1]:5d}")
        print()
        
        # Step 5: Check requirements
        print("Step 5: Checking Competition Requirements")
        print("-" * 70)
        
        requirements_met = True
        
        # Check F1 Score
        if metrics['f1_score'] >= 0.45:
            print(f"✓ F1 Score: {metrics['f1_score']:.4f} (Target: ≥ 0.45)")
        else:
            print(f"✗ F1 Score: {metrics['f1_score']:.4f} (Target: ≥ 0.45) - BELOW TARGET")
            requirements_met = False
        
        # Check Budget
        if oracle_fn.query_count[0] <= budget:
            print(f"✓ Budget: {oracle_fn.query_count[0]}/{budget} queries used")
        else:
            print(f"✗ Budget: {oracle_fn.query_count[0]}/{budget} - EXCEEDED")
            requirements_met = False
        
        # Check Runtime
        if runtime < 300:
            print(f"✓ Runtime: {runtime:.2f}s (Limit: < 300s)")
        else:
            print(f"✗ Runtime: {runtime:.2f}s (Limit: < 300s) - TOO SLOW")
            requirements_met = False
        
        print()
        print("=" * 70)
        if requirements_met:
            print("🎉 ALL REQUIREMENTS MET! Agent is ready for submission.")
        else:
            print("⚠️  Some requirements not met. Consider tuning parameters.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error running agent: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
