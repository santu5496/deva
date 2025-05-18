#!/usr/bin/env python3
"""
Download and prepare Kaggle datasets for leprosy detection.
Uses two datasets:
1. Skin disease dataset for leprosy positive samples
2. Skin types dataset for healthy skin (negative samples)
"""

import os
import shutil
import logging
import kagglehub
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset paths
DATASET_BASE_DIR = "dataset/leprosy_dataset"
POSITIVE_DIR = os.path.join(DATASET_BASE_DIR, "positive")
NEGATIVE_DIR = os.path.join(DATASET_BASE_DIR, "negative")
IRRELEVANT_DIR = os.path.join(DATASET_BASE_DIR, "irrelevant")

def setup_directories():
    """Create the necessary directory structure if it doesn't exist."""
    for directory in [DATASET_BASE_DIR, POSITIVE_DIR, NEGATIVE_DIR, IRRELEVANT_DIR]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created/verified: {directory}")

def download_skin_disease_dataset():
    """Download skin disease dataset for leprosy samples (positive)."""
    logger.info("Downloading skin disease dataset...")
    try:
        path = kagglehub.dataset_download("subirbiswas19/skin-disease-dataset")
        logger.info(f"Skin disease dataset downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to download skin disease dataset: {e}")
        return None

def download_healthy_skin_dataset():
    """Download healthy skin dataset for normal skin samples (negative)."""
    logger.info("Downloading healthy skin dataset...")
    try:
        path = kagglehub.dataset_download("shakyadissanayake/oily-dry-and-normal-skin-types-dataset")
        logger.info(f"Healthy skin dataset downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to download healthy skin dataset: {e}")
        return None

def process_skin_disease_dataset(dataset_path):
    """Extract leprosy images from the skin disease dataset and place in positive folder."""
    if not dataset_path:
        logger.error("Dataset path is empty, cannot process skin disease dataset")
        return
    
    # Find all leprosy related images in the dataset
    leprosy_dir = os.path.join(dataset_path, "Leprosy")
    if not os.path.exists(leprosy_dir):
        logger.warning(f"Leprosy directory not found at {leprosy_dir}")
        # Try to find leprosy images in the whole dataset
        for root, dirs, files in os.walk(dataset_path):
            if "leprosy" in root.lower():
                leprosy_dir = root
                logger.info(f"Found leprosy directory at: {leprosy_dir}")
                break
        else:
            logger.warning("Could not find a specific leprosy directory, searching for image files...")
            
            # Count of leprosy images copied
            leprosy_count = 0
            
            # Search for leprosy files by name pattern
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')) and 'leprosy' in file.lower():
                        source_path = os.path.join(root, file)
                        dest_path = os.path.join(POSITIVE_DIR, file)
                        shutil.copy(source_path, dest_path)
                        logger.info(f"Copied leprosy image: {file}")
                        leprosy_count += 1
            
            if leprosy_count == 0:
                logger.error("No leprosy images found in the dataset")
            else:
                logger.info(f"Copied {leprosy_count} leprosy images to {POSITIVE_DIR}")
            return
    
    # If we found a leprosy directory, copy all images from it
    copied_count = 0
    for root, dirs, files in os.walk(leprosy_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                source_path = os.path.join(root, file)
                dest_path = os.path.join(POSITIVE_DIR, file)
                shutil.copy(source_path, dest_path)
                copied_count += 1
    
    logger.info(f"Copied {copied_count} leprosy images to {POSITIVE_DIR}")

def process_healthy_skin_dataset(dataset_path):
    """Extract healthy skin images and place in negative folder."""
    if not dataset_path:
        logger.error("Dataset path is empty, cannot process healthy skin dataset")
        return
    
    # Count of healthy skin images copied
    healthy_count = 0
    
    # Look for normal/healthy skin directories
    for root, dirs, files in os.walk(dataset_path):
        if any(term in root.lower() for term in ['normal', 'healthy', 'regular']):
            logger.info(f"Found healthy skin directory: {root}")
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(NEGATIVE_DIR, file)
                    shutil.copy(source_path, dest_path)
                    healthy_count += 1
    
    # If we didn't find enough images in directories with specific names, 
    # grab all skin images (excluding those that might be diseases)
    if healthy_count < 10:
        logger.warning(f"Only found {healthy_count} healthy skin images in normal/healthy directories")
        logger.info("Searching for additional healthy skin images...")
        
        for root, dirs, files in os.walk(dataset_path):
            # Skip directories that might contain disease images
            if any(term in root.lower() for term in ['disease', 'leprosy', 'infection']):
                continue
                
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Skip files that might be disease images
                    if any(term in file.lower() for term in ['disease', 'leprosy', 'infection']):
                        continue
                        
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(NEGATIVE_DIR, file)
                    
                    # Check if we've already copied this file
                    if not os.path.exists(dest_path):
                        shutil.copy(source_path, dest_path)
                        healthy_count += 1
    
    logger.info(f"Copied {healthy_count} healthy skin images to {NEGATIVE_DIR}")

def check_dataset_balance():
    """Check how many images we have in each category and log the information."""
    positive_count = len([f for f in os.listdir(POSITIVE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    negative_count = len([f for f in os.listdir(NEGATIVE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    irrelevant_count = len([f for f in os.listdir(IRRELEVANT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  - Positive (leprosy) images: {positive_count}")
    logger.info(f"  - Negative (healthy skin) images: {negative_count}")
    logger.info(f"  - Irrelevant images: {irrelevant_count}")
    
    # Provide warning if dataset is very imbalanced
    if positive_count == 0 or negative_count == 0:
        logger.warning("One category has no images! The model will not train properly.")
    elif positive_count / negative_count < 0.2 or negative_count / positive_count < 0.2:
        logger.warning("Dataset is highly imbalanced. This may affect model performance.")

def main():
    """Main function to orchestrate the dataset download and processing."""
    logger.info("Starting Kaggle dataset download and processing for leprosy detection")
    
    # Set up directory structure
    setup_directories()
    
    # Download datasets
    skin_disease_path = download_skin_disease_dataset()
    healthy_skin_path = download_healthy_skin_dataset()
    
    # Process datasets
    if skin_disease_path:
        process_skin_disease_dataset(skin_disease_path)
    
    if healthy_skin_path:
        process_healthy_skin_dataset(healthy_skin_path)
    
    # Check dataset balance
    check_dataset_balance()
    
    # Create static samples directories
    static_samples_dir = "static/samples"
    for category in ["positive", "negative", "irrelevant"]:
        os.makedirs(os.path.join(static_samples_dir, category), exist_ok=True)
    
    # Copy a sample of images to the static samples directory
    for category in ["positive", "negative"]:
        source_dir = os.path.join(DATASET_BASE_DIR, category)
        dest_dir = os.path.join(static_samples_dir, category)
        
        files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for i, file in enumerate(files[:10]):  # Copy up to 10 files
            source_path = os.path.join(source_dir, file)
            dest_path = os.path.join(dest_dir, file)
            if not os.path.exists(dest_path):
                shutil.copy(source_path, dest_path)
                logger.info(f"Copied sample {file} to {dest_dir}")
    
    logger.info("Dataset download and processing complete")
    logger.info("Run 'python train_model.py' to train the model with the new dataset")

if __name__ == "__main__":
    main()