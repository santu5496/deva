#!/usr/bin/env python3
"""
Create and train a new model with the correct feature dimensions.
"""

import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model_utils import preprocess_image, extract_features

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get a sample image to determine feature dimensions
def get_feature_dimension():
    """Get feature dimension from a sample image"""
    sample_dirs = ['test_samples/positive', 'test_samples/negative']
    
    for directory in sample_dirs:
        if os.path.exists(directory):
            files = os.listdir(directory)
            if files:
                sample_path = os.path.join(directory, files[0])
                img = preprocess_image(sample_path)
                features = extract_features(img)
                return features.shape[1]
    
    # Default dimension if no samples found
    return 25

def train_new_model(output_path='model/leprosy_classifier.pkl', test_size=0.2):
    """
    Train a new model using the test samples
    
    Args:
        output_path (str): Path to save the model
        test_size (float): Proportion of data to use for testing
        
    Returns:
        model: Trained model
    """
    # Get feature dimension
    feature_dim = get_feature_dimension()
    logger.info(f"Feature dimension: {feature_dim}")
    
    # Prepare data
    X = []
    y = []
    
    # Process positive samples
    positive_dir = 'test_samples/positive'
    if os.path.exists(positive_dir):
        for filename in os.listdir(positive_dir):
            img_path = os.path.join(positive_dir, filename)
            if os.path.isfile(img_path):
                img = preprocess_image(img_path)
                features = extract_features(img)
                X.append(features[0])
                y.append(1)  # Positive class
    
    # Process negative samples
    negative_dir = 'test_samples/negative'
    if os.path.exists(negative_dir):
        for filename in os.listdir(negative_dir):
            img_path = os.path.join(negative_dir, filename)
            if os.path.isfile(img_path):
                img = preprocess_image(img_path)
                features = extract_features(img)
                X.append(features[0])
                y.append(0)  # Negative class
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=15,      # Maximum depth of trees
        min_samples_split=5,  # Minimum samples required to split a node
        random_state=42,   # For reproducibility
        n_jobs=-1,         # Use all available cores
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Train model
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    # Create classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("Classification Report:")
    if '1' in report:
        logger.info(f"Precision (Positive): {report['1']['precision']:.4f}")
        logger.info(f"Recall (Positive): {report['1']['recall']:.4f}")
        logger.info(f"F1-score (Positive): {report['1']['f1-score']:.4f}")
    else:
        logger.info("No positive samples in test set")
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {output_path}")
    
    # Save metadata
    metadata_path = output_path.replace('.pkl', '_meta.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None,
            'last_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'feature_dimension': feature_dim
        }, f)
    
    return model

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Train new model
    model = train_new_model()
    
    if model:
        logger.info("Model training completed successfully!")
    else:
        logger.error("Model training failed.")