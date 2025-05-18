#!/usr/bin/env python3
"""
Train the leprosy model with a subset of images for faster training.
This script loads a subset of the full dataset to speed up model training process.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import random
from PIL import Image
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model_utils import preprocess_image, extract_features

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_SAMPLES_PER_CLASS = 100  # Limit samples per class for faster training

def load_and_preprocess_subset(data_dir, img_size=(224, 224), max_per_class=MAX_SAMPLES_PER_CLASS):
    """
    Load and preprocess a subset of images from directory structure
    
    Args:
        data_dir (str): Directory containing class folders
        img_size (tuple): Target image size
        max_per_class (int): Maximum number of samples per class to use
        
    Returns:
        features, labels: Processed features and corresponding labels
    """
    # Check if directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} not found")
        return [], []
    
    # Find class directories (positive and negative)
    positive_dir = os.path.join(data_dir, 'positive')
    negative_dir = os.path.join(data_dir, 'negative')
    
    if not os.path.exists(positive_dir) or not os.path.exists(negative_dir):
        logger.error(f"Missing class directories in {data_dir}")
        return [], []
    
    # Load and process images
    all_features = []
    all_labels = []
    
    # Process positive samples
    positive_files = glob.glob(os.path.join(positive_dir, '*.*'))
    logger.info(f"Found {len(positive_files)} positive samples (using max {max_per_class})")
    
    # Randomly select subset of positive files
    if len(positive_files) > max_per_class:
        positive_files = random.sample(positive_files, max_per_class)
    
    for img_path in positive_files:
        try:
            # Preprocess image
            img = preprocess_image(img_path, target_size=img_size)
            
            # Extract features
            features = extract_features(img)
            
            # Add to dataset
            all_features.append(features[0])
            all_labels.append(1)  # Positive class
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
    
    # Process negative samples
    negative_files = glob.glob(os.path.join(negative_dir, '*.*'))
    logger.info(f"Found {len(negative_files)} negative samples (using max {max_per_class})")
    
    # Randomly select subset of negative files
    if len(negative_files) > max_per_class:
        negative_files = random.sample(negative_files, max_per_class)
    
    for img_path in negative_files:
        try:
            # Preprocess image
            img = preprocess_image(img_path, target_size=img_size)
            
            # Extract features
            features = extract_features(img)
            
            # Add to dataset
            all_features.append(features[0])
            all_labels.append(0)  # Negative class
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    logger.info(f"Processed dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y

def train_model_subset(data_dir='dataset/leprosy_dataset', 
                  img_size=(224, 224),
                  output_path='model/leprosy_classifier.pkl',
                  max_samples=MAX_SAMPLES_PER_CLASS):
    """
    Train a model for leprosy detection using a subset of data
    
    Args:
        data_dir (str): Directory containing the dataset
        img_size (tuple): Target image size
        output_path (str): Path to save the trained model
        max_samples (int): Maximum number of samples per class
        
    Returns:
        model: Trained model
    """
    # Load and preprocess subset of data
    logger.info("Loading and preprocessing data subset...")
    X, y = load_and_preprocess_subset(data_dir, img_size, max_samples)
    
    if len(X) == 0 or len(y) == 0:
        logger.error("No valid samples found for training.")
        return None
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create model
    logger.info("Creating model...")
    # Get feature dimension from first sample
    feature_dim = X_train.shape[1]
    logger.info(f"Feature dimension: {feature_dim}")
    
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
    
    # Create a report
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("Classification Report:")
    logger.info(f"Precision (Positive): {report.get('1', {}).get('precision', 0):.4f}")
    logger.info(f"Recall (Positive): {report.get('1', {}).get('recall', 0):.4f}")
    logger.info(f"F1-score (Positive): {report.get('1', {}).get('f1-score', 0):.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    
    # Add labels to the plot
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_path = os.path.join(os.path.dirname(output_path), 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {plot_path}")
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {output_path}")
    
    # Save feature importance plot
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        feature_importance = model.feature_importances_
        # Sort feature importance in descending order
        sorted_idx = np.argsort(feature_importance)[::-1]
        pos = np.arange(min(20, len(sorted_idx)))  # Show top 20 features
        plt.barh(pos, feature_importance[sorted_idx][:20])
        plt.title('Feature Importance (Top 20)')
        plt.tight_layout()
        
        importance_path = os.path.join(os.path.dirname(output_path), 'feature_importance.png')
        plt.savefig(importance_path)
        plt.close()
        logger.info(f"Feature importance plot saved to {importance_path}")
    
    # Save metadata
    metadata_path = output_path.replace('.pkl', '_meta.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None,
            'last_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'subset_training': True,
            'max_samples_per_class': max_samples
        }, f)
    
    return model

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Train model with subset of data
    model = train_model_subset(
        data_dir='dataset/leprosy_dataset',
        output_path='model/leprosy_classifier.pkl',
        max_samples=MAX_SAMPLES_PER_CLASS
    )
    
    if model:
        logger.info("Model training completed successfully!")
    else:
        logger.error("Model training failed. Check logs for details.")