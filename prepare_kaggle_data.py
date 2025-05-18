#!/usr/bin/env python3
"""
Prepare Kaggle datasets for leprosy detection model training.
This script processes the downloaded Kaggle datasets and creates a structured dataset
for binary classification (leprosy vs. healthy skin).
"""

import os
import sys
import glob
import shutil
import logging
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

# Paths to Kaggle datasets
SKIN_DISEASE_PATH = "/home/runner/.cache/kagglehub/datasets/subirbiswas19/skin-disease-dataset/versions/1/skin-disease-datasaet"
HEALTHY_SKIN_PATH = "/home/runner/.cache/kagglehub/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset/versions/1"

# For the skin disease dataset, we'll treat these diseases as similar to leprosy (for demonstration)
# In a real clinical setting, you would want actual leprosy images
LEPROSY_SIMILAR_CONDITIONS = [
    "BA- cellulitis",  # Bacterial skin infection with redness/swelling
    "BA-impetigo",     # Bacterial skin infection with blisters
    "FU-ringworm",     # Fungal infection with ring-shaped rash
    "PA-cutaneous-larva-migrans"  # Parasitic infection with serpiginous tracks
]

def setup_directories():
    """Create the necessary directory structure if it doesn't exist."""
    for directory in [DATASET_BASE_DIR, POSITIVE_DIR, NEGATIVE_DIR, IRRELEVANT_DIR]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created/verified: {directory}")

def copy_leprosy_similar_images():
    """Copy images from conditions similar to leprosy as positive samples."""
    count = 0
    
    # Look in both train and test sets
    for dataset_type in ["train_set", "test_set"]:
        dataset_path = os.path.join(SKIN_DISEASE_PATH, dataset_type)
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            continue
            
        # Process each disease condition that is similar to leprosy
        for condition in LEPROSY_SIMILAR_CONDITIONS:
            condition_path = os.path.join(dataset_path, condition)
            if not os.path.exists(condition_path):
                logger.warning(f"Condition directory not found: {condition_path}")
                continue
                
            # Copy all images from this condition to positive directory
            for img_file in glob.glob(os.path.join(condition_path, "*.jpg")):
                # Create a unique filename to avoid overwriting
                base_name = os.path.basename(img_file)
                condition_prefix = condition.replace(" ", "_").replace("-", "_")
                new_filename = f"{condition_prefix}_{base_name}"
                dest_path = os.path.join(POSITIVE_DIR, new_filename)
                
                # Copy the file if it doesn't already exist
                if not os.path.exists(dest_path):
                    shutil.copy(img_file, dest_path)
                    count += 1
                    
                    # Log progress every 20 files
                    if count % 20 == 0:
                        logger.info(f"Copied {count} positive images so far...")
    
    logger.info(f"Total positive (leprosy-similar) images copied: {count}")

def copy_healthy_skin_images():
    """Copy normal skin images as negative samples."""
    count = 0
    
    # Look for normal skin images in the dataset
    normal_dirs = []
    
    # Find all directories containing normal skin images
    for root, dirs, files in os.walk(HEALTHY_SKIN_PATH):
        if "normal" in root.lower():
            normal_dirs.append(root)
    
    if not normal_dirs:
        logger.warning("No normal skin directories found")
        return
        
    # Process each normal skin directory
    for normal_dir in normal_dirs:
        logger.info(f"Processing normal skin directory: {normal_dir}")
        
        # Copy all images from this directory to negative directory
        for img_ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_file in glob.glob(os.path.join(normal_dir, img_ext)):
                # Create a unique filename to avoid overwriting
                base_name = os.path.basename(img_file)
                dir_prefix = os.path.basename(normal_dir).replace(" ", "_").replace("-", "_")
                new_filename = f"{dir_prefix}_{base_name}"
                dest_path = os.path.join(NEGATIVE_DIR, new_filename)
                
                # Copy the file if it doesn't already exist
                if not os.path.exists(dest_path):
                    shutil.copy(img_file, dest_path)
                    count += 1
                    
                    # Log progress every 20 files
                    if count % 20 == 0:
                        logger.info(f"Copied {count} negative images so far...")
    
    logger.info(f"Total negative (healthy skin) images copied: {count}")

def copy_irrelevant_images():
    """Copy some non-skin images to the irrelevant category for testing."""
    # Use some disease images that are visually very different from leprosy
    non_skin_conditions = ["FU-nail-fungus", "VI-chickenpox", "VI-shingles"]
    count = 0
    
    # Look in both train and test sets
    for dataset_type in ["train_set", "test_set"]:
        dataset_path = os.path.join(SKIN_DISEASE_PATH, dataset_type)
        if not os.path.exists(dataset_path):
            continue
            
        # Process each non-skin condition
        for condition in non_skin_conditions:
            condition_path = os.path.join(dataset_path, condition)
            if not os.path.exists(condition_path):
                continue
                
            # Copy a sample of images (up to 10 per condition)
            image_files = glob.glob(os.path.join(condition_path, "*.jpg"))
            for img_file in image_files[:10]:  # Limit to 10 images per condition
                # Create a unique filename to avoid overwriting
                base_name = os.path.basename(img_file)
                condition_prefix = condition.replace(" ", "_").replace("-", "_")
                new_filename = f"{condition_prefix}_{base_name}"
                dest_path = os.path.join(IRRELEVANT_DIR, new_filename)
                
                # Copy the file if it doesn't already exist
                if not os.path.exists(dest_path):
                    shutil.copy(img_file, dest_path)
                    count += 1
    
    logger.info(f"Total irrelevant images copied: {count}")

def copy_to_static_samples():
    """Copy a sample of images to the static samples directory for the web interface."""
    static_samples_dir = "static/samples"
    for category in ["positive", "negative", "irrelevant"]:
        os.makedirs(os.path.join(static_samples_dir, category), exist_ok=True)
        
    # Copy up to 10 images from each category
    for category in ["positive", "negative", "irrelevant"]:
        source_dir = os.path.join(DATASET_BASE_DIR, category)
        dest_dir = os.path.join(static_samples_dir, category)
        
        # Get all image files
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(glob.glob(os.path.join(source_dir, ext)))
        
        # Sort and limit to 10 files
        image_files.sort()
        image_files = image_files[:10]
        
        # Copy files
        for img_file in image_files:
            base_name = os.path.basename(img_file)
            dest_path = os.path.join(dest_dir, base_name)
            if not os.path.exists(dest_path):
                shutil.copy(img_file, dest_path)
                logger.info(f"Copied {base_name} to {dest_dir}")
    
    logger.info("Sample images copied to static/samples directory")

def check_dataset_stats():
    """Check and report dataset statistics."""
    positive_count = len(glob.glob(os.path.join(POSITIVE_DIR, "*.jpg")))
    negative_count = len(glob.glob(os.path.join(NEGATIVE_DIR, "*.jpg")))
    irrelevant_count = len(glob.glob(os.path.join(IRRELEVANT_DIR, "*.jpg")))
    
    # Add other image formats
    for ext in ["*.jpeg", "*.png"]:
        positive_count += len(glob.glob(os.path.join(POSITIVE_DIR, ext)))
        negative_count += len(glob.glob(os.path.join(NEGATIVE_DIR, ext)))
        irrelevant_count += len(glob.glob(os.path.join(IRRELEVANT_DIR, ext)))
    
    logger.info("Dataset statistics:")
    logger.info(f"  - Positive (leprosy-like) images: {positive_count}")
    logger.info(f"  - Negative (healthy skin) images: {negative_count}")
    logger.info(f"  - Irrelevant images: {irrelevant_count}")
    
    total_images = positive_count + negative_count + irrelevant_count
    logger.info(f"  - Total images: {total_images}")
    
    # Check for potential issues
    if positive_count == 0 or negative_count == 0:
        logger.warning("One of the categories has no images! Model training will fail.")
    elif positive_count < 10 or negative_count < 10:
        logger.warning("Very few images in one category. Model may not train well.")
    
    return positive_count, negative_count, irrelevant_count

def main():
    """Main function to orchestrate the dataset preparation."""
    logger.info("Starting Kaggle dataset preparation for leprosy detection")
    
    # Set up directory structure
    setup_directories()
    
    # Copy images from skin disease dataset (leprosy-like conditions)
    copy_leprosy_similar_images()
    
    # Copy healthy skin images
    copy_healthy_skin_images()
    
    # Copy irrelevant images
    copy_irrelevant_images()
    
    # Check dataset statistics
    positive_count, negative_count, irrelevant_count = check_dataset_stats()
    
    # Copy samples to static directory for web interface
    copy_to_static_samples()
    
    logger.info("Dataset preparation complete")
    logger.info(f"Run 'python train_model.py' to train the model with {positive_count + negative_count} images")

if __name__ == "__main__":
    main()