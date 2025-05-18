#!/usr/bin/env python3
"""
Prepare test samples for evaluating the leprosy detection model.
This script copies 15 positive and 15 negative images to a test directory.
"""

import os
import random
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NUM_SAMPLES_PER_CLASS = 15
TEST_DIR = 'test_samples'
POSITIVE_TEST_DIR = os.path.join(TEST_DIR, 'positive')
NEGATIVE_TEST_DIR = os.path.join(TEST_DIR, 'negative')

def setup_test_directories():
    """Set up test directories"""
    for directory in [TEST_DIR, POSITIVE_TEST_DIR, NEGATIVE_TEST_DIR]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def get_image_files(directory):
    """Get list of image files in a directory"""
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return []
    
    # Get all files with common image extensions
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(directory) 
                  if os.path.isfile(os.path.join(directory, f)) and 
                  f.lower().endswith(image_extensions)]
    
    return image_files

def copy_sample_images():
    """Copy sample images to test directories"""
    # Source directories
    positive_dir = 'dataset/leprosy_dataset/positive'
    negative_dir = 'dataset/leprosy_dataset/negative'
    
    # Get image files
    positive_images = get_image_files(positive_dir)
    negative_images = get_image_files(negative_dir)
    
    logger.info(f"Found {len(positive_images)} positive images")
    logger.info(f"Found {len(negative_images)} negative images")
    
    # Randomly select samples
    selected_positive = random.sample(positive_images, 
                                     min(NUM_SAMPLES_PER_CLASS, len(positive_images)))
    selected_negative = random.sample(negative_images, 
                                     min(NUM_SAMPLES_PER_CLASS, len(negative_images)))
    
    # Copy positive images
    for i, filename in enumerate(selected_positive, 1):
        src_path = os.path.join(positive_dir, filename)
        dst_path = os.path.join(POSITIVE_TEST_DIR, f"positive_{i}_{filename}")
        shutil.copy(src_path, dst_path)
        logger.info(f"Copied positive image {i}: {filename}")
    
    # Copy negative images
    for i, filename in enumerate(selected_negative, 1):
        src_path = os.path.join(negative_dir, filename)
        dst_path = os.path.join(NEGATIVE_TEST_DIR, f"negative_{i}_{filename}")
        shutil.copy(src_path, dst_path)
        logger.info(f"Copied negative image {i}: {filename}")
    
    logger.info(f"Copied {len(selected_positive)} positive and {len(selected_negative)} negative images")

def main():
    """Main function"""
    logger.info("Preparing test samples...")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Set up directories
    setup_test_directories()
    
    # Copy sample images
    copy_sample_images()
    
    logger.info("Test sample preparation complete")

if __name__ == "__main__":
    main()