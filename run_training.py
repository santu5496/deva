#!/usr/bin/env python3
"""
This script trains the leprosy detection model using datasets from various sources.
It first attempts to download a dataset from Kaggle if credentials are available,
then falls back to sample images if needed.
"""

import os
import argparse
import logging
import time
from prepare_dataset import setup_data_directory, download_kaggle_dataset
from train_model import train_leprosy_model
from model_utils import train_model_from_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train the leprosy detection model')
    parser.add_argument('--img-size', type=int, default=224, help='Image size for training')
    parser.add_argument('--output', type=str, default='model/leprosy_classifier.pkl', help='Output path for the trained model')
    parser.add_argument('--data-dir', type=str, default=None, help='Path to the dataset directory (if already downloaded)')
    parser.add_argument('--no-kaggle', action='store_true', help='Skip attempting to download dataset from Kaggle')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Set log level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print training configuration
    logger.info("Training configuration:")
    logger.info(f"  Image size: {args.img_size}")
    logger.info(f"  Output path: {args.output}")
    
    # Prepare dataset directory
    start_time = time.time()
    
    # Get dataset
    if args.data_dir and os.path.exists(args.data_dir):
        logger.info(f"Using provided dataset directory: {args.data_dir}")
        dataset_path = args.data_dir
    elif not args.no_kaggle:
        # Try to get dataset from Kaggle
        logger.info("Attempting to download leprosy dataset from Kaggle...")
        dataset_path = download_kaggle_dataset()
        
        if dataset_path:
            logger.info(f"Using dataset from Kaggle: {dataset_path}")
        else:
            logger.warning("Failed to download dataset from Kaggle")
            # Set up a minimal dataset with sample images
            dataset_path = setup_data_directory()
    else:
        # Set up a minimal dataset with sample images
        dataset_path = setup_data_directory()
    
    # Check if dataset directories exist and contain files
    positive_dir = os.path.join(dataset_path, 'positive')
    negative_dir = os.path.join(dataset_path, 'negative')
    
    if not (os.path.exists(positive_dir) and os.path.exists(negative_dir)):
        logger.error("Dataset directories are missing. Please check the dataset structure.")
        return
    
    positive_count = len([f for f in os.listdir(positive_dir) if os.path.isfile(os.path.join(positive_dir, f))])
    negative_count = len([f for f in os.listdir(negative_dir) if os.path.isfile(os.path.join(negative_dir, f))])
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  Positive examples: {positive_count}")
    logger.info(f"  Negative examples: {negative_count}")
    
    if positive_count == 0 or negative_count == 0:
        logger.error("Both positive and negative examples are required for training")
        return
    
    # Train the model
    logger.info("Starting model training...")
    
    # Use train_model_from_dataset for direct training
    model = train_model_from_dataset(
        dataset_dir=dataset_path,
        output_path=args.output
    )
    
    # Also track training time
    training_time = time.time() - start_time
    
    # Get model metadata
    metadata = model.get_metadata() if hasattr(model, 'get_metadata') else {}
    
    # Print training results
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    logger.info(f"Model saved to: {args.output}")
    
    if 'accuracy' in metadata:
        logger.info(f"Model accuracy: {metadata['accuracy']:.4f}")
    
    if 'training_samples' in metadata:
        logger.info(f"Training samples: {metadata['training_samples']}")
    
    logger.info("You can now use the model for real-time predictions!")
    return model

if __name__ == "__main__":
    main()