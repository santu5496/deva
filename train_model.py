import os
import glob
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from kaggle_utils import setup_kaggle_credentials, download_dataset
from model_utils import preprocess_image, extract_features

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model():
    """
    Create a machine learning model for skin lesion classification
    
    Returns:
        model: Initialized scikit-learn model
    """
    # Create a RandomForest classifier (robust and effective for many tasks)
    model = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=15,      # Maximum depth of trees
        min_samples_split=5,  # Minimum samples required to split a node
        random_state=42,   # For reproducibility
        n_jobs=-1,         # Use all available cores
        class_weight='balanced'  # Handle class imbalance
    )
    
    return model

def load_and_preprocess_images(data_dir, img_size=(224, 224)):
    """
    Load and preprocess images from directory structure
    
    Args:
        data_dir (str): Directory containing class folders
        img_size (tuple): Target image size
        
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
    logger.info(f"Found {len(positive_files)} positive samples")
    
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
    logger.info(f"Found {len(negative_files)} negative samples")
    
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

def train_leprosy_model(data_dir='dataset/leprosy_dataset', 
                        img_size=(224, 224),
                        output_path='model/leprosy_classifier.pkl'):
    """
    Train a model for leprosy detection using scikit-learn
    
    Args:
        data_dir (str): Directory containing the dataset
        img_size (tuple): Target image size
        output_path (str): Path to save the trained model
        
    Returns:
        history_dict: Training history
        model: Trained model
    """
    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} not found.")
        return None, None
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    X, y = load_and_preprocess_images(data_dir, img_size)
    
    if len(X) == 0 or len(y) == 0:
        logger.error("No valid samples found for training.")
        return None, None
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_model()
    
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
    history_dict = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    }
    
    metadata_path = output_path.replace('.pkl', '_meta.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None,
            'last_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'accuracy': accuracy,
            'training_samples': len(X_train)
        }, f)
    
    return history_dict, model

def prepare_leprosy_dataset():
    """
    Use the already prepared dataset from prepare_kaggle_data.py
    
    Returns:
        str: Path to the prepared dataset directory
    """
    # The dataset has already been prepared by prepare_kaggle_data.py
    dataset_path = "dataset/leprosy_dataset"
    
    if os.path.exists(dataset_path):
        # Count images in each category
        positive_dir = os.path.join(dataset_path, "positive")
        negative_dir = os.path.join(dataset_path, "negative")
        irrelevant_dir = os.path.join(dataset_path, "irrelevant")
        
        if os.path.exists(positive_dir) and os.path.exists(negative_dir):
            positive_count = len([f for f in os.listdir(positive_dir) 
                                if os.path.isfile(os.path.join(positive_dir, f))])
            negative_count = len([f for f in os.listdir(negative_dir) 
                                if os.path.isfile(os.path.join(negative_dir, f))])
            
            logger.info(f"Found {positive_count} positive samples and {negative_count} negative samples in {dataset_path}")
            return dataset_path
    
    logger.warning("Prepared dataset not found. Will try to create a new one.")
    
    # If we get here, the dataset wasn't prepared properly
    try:
        # Try running the prepare_kaggle_data.py script
        import prepare_kaggle_data
        prepare_kaggle_data.main()
        logger.info("Successfully prepared dataset from Kaggle datasets")
        return dataset_path
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        return None

def setup_data_directory():
    """
    Set up the data directory structure for a binary classification problem:
    - dataset/leprosy_dataset/positive
    - dataset/leprosy_dataset/negative
    
    This is needed if the downloaded dataset has a different structure
    
    Returns:
        str: Path to the prepared dataset directory
    """
    # Create directory structure
    base_dir = "dataset/leprosy_dataset"
    positive_dir = os.path.join(base_dir, "positive")
    negative_dir = os.path.join(base_dir, "negative")
    
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)
    
    # If no datasets were downloaded, create a minimal sample dataset for testing
    if not os.listdir(positive_dir) and not os.listdir(negative_dir):
        logger.warning("No dataset available. Please download a real dataset for proper training.")
        
    return base_dir

if __name__ == "__main__":
    # Prepare dataset
    logger.info("Preparing leprosy dataset...")
    dataset_path = prepare_leprosy_dataset()
    
    if not dataset_path:
        logger.info("Setting up minimal dataset directory structure...")
        dataset_path = setup_data_directory()
    
    # Train model
    logger.info("Starting model training...")
    history, model = train_leprosy_model(
        data_dir=dataset_path,
        output_path='model/leprosy_classifier.pkl'
    )
    
    if history and model:
        logger.info("Model training completed successfully!")
    else:
        logger.error("Model training failed. Check logs for details.")