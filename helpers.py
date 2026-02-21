"""
Helper functions for the Active Learning Agent
(Optional - can be used to extend functionality)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def calculate_uncertainty(predictions_proba: np.ndarray) -> np.ndarray:
    """
    Calculate uncertainty scores for predictions.
    Higher uncertainty = less confident predictions.
    
    Args:
        predictions_proba: Probability predictions from classifier
    
    Returns:
        np.ndarray: Uncertainty scores
    """
    # Use entropy as uncertainty measure
    proba_sorted = np.sort(predictions_proba, axis=1)
    # Margin sampling: difference between top 2 classes
    margin = proba_sorted[:, -1] - proba_sorted[:, -2]
    return 1 - margin


def diversity_sampling(X: np.ndarray, n_samples: int, random_state: int = 42) -> List[int]:
    """
    Select diverse samples using k-means++ initialization strategy.
    
    Args:
        X: Feature matrix
        n_samples: Number of samples to select
        random_state: Random seed
    
    Returns:
        List[int]: Indices of selected samples
    """
    np.random.seed(random_state)
    n_total = X.shape[0]
    
    # Start with random sample
    selected = [np.random.randint(n_total)]
    
    for _ in range(n_samples - 1):
        # Calculate minimum distance to already selected samples
        distances = np.min([np.linalg.norm(X - X[idx], axis=1) for idx in selected], axis=0)
        # Select point with maximum minimum distance
        next_idx = np.argmax(distances)
        selected.append(next_idx)
    
    return selected


def evaluate_performance(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate performance metrics for fraud detection.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        dict: Dictionary with precision, recall, f1, and accuracy
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred)
    }


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess features if needed (handle missing values, outliers, etc.)
    
    Args:
        df: Raw feature DataFrame
    
    Returns:
        pd.DataFrame: Preprocessed features
    """
    # Handle missing values by filling with median
    df_clean = df.fillna(df.median())
    
    # Cap extreme outliers at 99th percentile
    for col in df_clean.columns:
        percentile_99 = df_clean[col].quantile(0.99)
        df_clean[col] = df_clean[col].clip(upper=percentile_99)
    
    return df_clean


def enhance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create high-signal compound features for fraud detection.
    These features amplify fraud patterns by combining multiple risk signals.
    
    Strategy: "Risk Concentrator" - multiply signals rather than add them.
    A candidate with multiple red flags has exponentially higher fraud risk.
    
    Args:
        df: Raw feature DataFrame
    
    Returns:
        pd.DataFrame: Enhanced feature DataFrame with new columns
    """
    df_enhanced = df.copy()
    
    # Pillar A: Total Risk Score
    # Combines independent risk signals from IP, email, and institution
    # High values indicate multi-dimensional suspicious activity
    df_enhanced['total_risk'] = (
        df['ip_risk_score'] + 
        df['email_risk_score'] + 
        df['institution_risk_score']
    )
    
    # Pillar B: Bot Index
    # Identifies automated behavior: high application frequency + copy-paste patterns
    # Real candidates rarely apply 7+ times/week while copy-pasting everything
    df_enhanced['bot_index'] = df['applications_7d'] * df['copy_paste_ratio']
    
    # Pillar C: Academic Red Flag
    # Detects credential inflation: high GPA anomaly vs. limited experience
    # Junior candidates with suspicious academic metrics = likely fake profiles
    df_enhanced['academic_red_flag'] = (
        df['gpa_anomaly_score'] / (df['experience_years'] + 1)
    )
    
    # Pillar D: Velocity Risk
    # Combines login velocity with failed logins
    # High login attempts + failures = potential account takeover or bot
    df_enhanced['velocity_risk'] = (
        df['login_velocity_24h'] * (df['failed_logins_24h'] + 1)
    )
    
    # Pillar E: Profile Maturity Mismatch
    # New profiles with excessive applications = spam/fraud pattern
    # Divide by profile_age_days to catch recently created aggressive profiles
    df_enhanced['profile_spam_score'] = (
        df['applications_30d'] / (df['profile_age_days'] + 1)
    )
    
    return df_enhanced
