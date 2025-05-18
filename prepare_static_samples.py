#!/usr/bin/env python3
"""
Prepare static sample images for the web interface.
This script copies a few images from test_samples to static/samples for display on the samples page.
"""

import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
STATIC_SAMPLES_DIR = 'static/samples'
POSITIVE_STATIC_DIR = os.path.join(STATIC_SAMPLES_DIR, 'positive')
NEGATIVE_STATIC_DIR = os.path.join(STATIC_SAMPLES_DIR, 'negative')
IRRELEVANT_STATIC_DIR = os.path.join(STATIC_SAMPLES_DIR, 'irrelevant')

TEST_SAMPLES_DIR = 'test_samples'
POSITIVE_TEST_DIR = os.path.join(TEST_SAMPLES_DIR, 'positive')
NEGATIVE_TEST_DIR = os.path.join(TEST_SAMPLES_DIR, 'negative')

def setup_directories():
    """Create the necessary directory structure if it doesn't exist."""
    for directory in [STATIC_SAMPLES_DIR, POSITIVE_STATIC_DIR, NEGATIVE_STATIC_DIR, IRRELEVANT_STATIC_DIR]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def copy_test_samples_to_static():
    """Copy test samples to static samples directory."""
    # Check if test samples exist
    if not os.path.exists(POSITIVE_TEST_DIR) or not os.path.exists(NEGATIVE_TEST_DIR):
        logger.warning("Test samples not found. Run prepare_test_samples.py first.")
        return False
    
    # Get list of files
    positive_files = [f for f in os.listdir(POSITIVE_TEST_DIR) 
                     if os.path.isfile(os.path.join(POSITIVE_TEST_DIR, f))]
    negative_files = [f for f in os.listdir(NEGATIVE_TEST_DIR) 
                     if os.path.isfile(os.path.join(NEGATIVE_TEST_DIR, f))]
    
    # Limit to 10 files per category
    positive_files = positive_files[:10]
    negative_files = negative_files[:10]
    
    # Copy positive files
    logger.info(f"Copying {len(positive_files)} positive images...")
    for filename in positive_files:
        src_path = os.path.join(POSITIVE_TEST_DIR, filename)
        dest_path = os.path.join(POSITIVE_STATIC_DIR, filename)
        shutil.copy(src_path, dest_path)
        logger.info(f"Copied {filename}")
    
    # Copy negative files
    logger.info(f"Copying {len(negative_files)} negative images...")
    for filename in negative_files:
        src_path = os.path.join(NEGATIVE_TEST_DIR, filename)
        dest_path = os.path.join(NEGATIVE_STATIC_DIR, filename)
        shutil.copy(src_path, dest_path)
        logger.info(f"Copied {filename}")
    
    # Create a couple of irrelevant images for testing
    # We'll just copy a few images from positive/negative and rename them
    logger.info("Creating irrelevant samples...")
    if positive_files and negative_files:
        src_paths = [
            os.path.join(POSITIVE_TEST_DIR, positive_files[0]),
            os.path.join(NEGATIVE_TEST_DIR, negative_files[0])
        ]
        for i, src_path in enumerate(src_paths):
            dest_filename = f"irrelevant_{i+1}_{os.path.basename(src_path)}"
            dest_path = os.path.join(IRRELEVANT_STATIC_DIR, dest_filename)
            shutil.copy(src_path, dest_path)
            logger.info(f"Created irrelevant sample: {dest_filename}")

    return True

def main():
    """Main function to prepare static samples."""
    logger.info("Preparing static sample images...")
    
    # Set up directories
    setup_directories()
    
    # Copy test samples to static samples
    success = copy_test_samples_to_static()
    
    if success:
        logger.info("Static samples prepared successfully.")
    else:
        logger.warning("Failed to prepare static samples.")

if __name__ == "__main__":
    main()