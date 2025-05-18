#!/usr/bin/env python3
"""
Setup complete project structure for the Leprosy Detection AI Application.
This script creates all required directories and copies necessary files for local development.
"""

import os
import sys
import shutil
import logging
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define all required directories
DIRECTORIES = [
    # Main directories
    'model',
    'dataset',
    'dataset/leprosy_dataset',
    'dataset/leprosy_dataset/positive',
    'dataset/leprosy_dataset/negative',
    'dataset/leprosy_dataset/irrelevant',
    'test_samples',
    'test_samples/positive',
    'test_samples/negative',
    'test_results',
    
    # Static directories
    'static',
    'static/css',
    'static/js',
    'static/images',
    'static/uploads',
    'static/samples',
    'static/samples/positive',
    'static/samples/negative',
    'static/samples/irrelevant',
    
    # Templates directory
    'templates',
    
    # Instance directory (for SQLite database)
    'instance',
    
    # History related directories
    'static/history',
    'static/history/images',
    'static/history/results'
]

def create_directory_structure():
    """Create all required directories"""
    for directory in DIRECTORIES:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True

def copy_test_samples_to_static():
    """Copy test samples to static directory for the samples page"""
    # Source directories
    positive_test_dir = 'test_samples/positive'
    negative_test_dir = 'test_samples/negative'
    
    # Destination directories
    positive_static_dir = 'static/samples/positive'
    negative_static_dir = 'static/samples/negative'
    irrelevant_static_dir = 'static/samples/irrelevant'
    
    # Check if test samples exist
    if (os.path.exists(positive_test_dir) and os.listdir(positive_test_dir) and
        os.path.exists(negative_test_dir) and os.listdir(negative_test_dir)):
        
        # Get files
        positive_files = [f for f in os.listdir(positive_test_dir) 
                        if os.path.isfile(os.path.join(positive_test_dir, f))]
        negative_files = [f for f in os.listdir(negative_test_dir) 
                        if os.path.isfile(os.path.join(negative_test_dir, f))]
        
        # Limit to 10 files per category
        positive_files = positive_files[:10]
        negative_files = negative_files[:10]
        
        # Copy positive files
        for filename in positive_files:
            src = os.path.join(positive_test_dir, filename)
            dest = os.path.join(positive_static_dir, filename)
            shutil.copy(src, dest)
            logger.info(f"Copied to static samples: {filename}")
        
        # Copy negative files
        for filename in negative_files:
            src = os.path.join(negative_test_dir, filename)
            dest = os.path.join(negative_static_dir, filename)
            shutil.copy(src, dest)
            logger.info(f"Copied to static samples: {filename}")
        
        # Create a few irrelevant samples (for testing)
        if positive_files and negative_files:
            # Just copy one of each and rename
            src_files = [
                os.path.join(positive_test_dir, positive_files[0]),
                os.path.join(negative_test_dir, negative_files[0])
            ]
            
            for i, src in enumerate(src_files):
                filename = os.path.basename(src)
                dest = os.path.join(irrelevant_static_dir, f"irrelevant_{i+1}_{filename}")
                shutil.copy(src, dest)
                logger.info(f"Created irrelevant sample: {os.path.basename(dest)}")
        
        return True
    else:
        logger.warning("Test samples not found or empty. Skipping static sample creation.")
        return False

def create_dummy_history_entries():
    """Create a few dummy history entries for testing"""
    # Create history directories if they don't exist
    history_img_dir = 'static/history/images'
    history_results_dir = 'static/history/results'
    os.makedirs(history_img_dir, exist_ok=True)
    os.makedirs(history_results_dir, exist_ok=True)
    
    # Check if we have test samples to use
    positive_test_dir = 'test_samples/positive'
    negative_test_dir = 'test_samples/negative'
    
    if (os.path.exists(positive_test_dir) and os.listdir(positive_test_dir) and
        os.path.exists(negative_test_dir) and os.listdir(negative_test_dir)):
        
        # Get one positive and one negative sample
        positive_files = [f for f in os.listdir(positive_test_dir) 
                        if os.path.isfile(os.path.join(positive_test_dir, f))]
        negative_files = [f for f in os.listdir(negative_test_dir) 
                        if os.path.isfile(os.path.join(negative_test_dir, f))]
        
        if positive_files and negative_files:
            # Copy one of each with timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Positive sample
            pos_src = os.path.join(positive_test_dir, positive_files[0])
            pos_filename = f"1_{timestamp}_history_positive_{positive_files[0]}"
            pos_dest = os.path.join(history_img_dir, pos_filename)
            shutil.copy(pos_src, pos_dest)
            logger.info(f"Created history image: {pos_filename}")
            
            # Negative sample
            neg_src = os.path.join(negative_test_dir, negative_files[0])
            neg_filename = f"1_{timestamp}_history_negative_{negative_files[0]}"
            neg_dest = os.path.join(history_img_dir, neg_filename)
            shutil.copy(neg_src, neg_dest)
            logger.info(f"Created history image: {neg_filename}")
            
            return True
    
    logger.warning("Could not create history samples due to missing test samples.")
    return False

def create_placeholder_files():
    """Create placeholder files to keep directory structure in git"""
    # Create .gitkeep files in all empty directories
    for directory in DIRECTORIES:
        if os.path.exists(directory) and not os.listdir(directory):
            with open(os.path.join(directory, '.gitkeep'), 'w') as f:
                pass
            logger.info(f"Created .gitkeep in {directory}")
    
    return True

def main():
    """Main function to setup project structure"""
    logger.info("Setting up project structure...")
    
    # Create directory structure
    create_directory_structure()
    
    # Create placeholder files
    create_placeholder_files()
    
    # Copy test samples to static directory
    if os.path.exists('test_samples/positive') and os.path.exists('test_samples/negative'):
        copy_test_samples_to_static()
    
    # Create dummy history entries for testing
    create_dummy_history_entries()
    
    logger.info("Project structure setup complete!")
    logger.info("\nNext steps:")
    logger.info("1. Run the application: python main.py")
    logger.info("2. Access the application at: http://127.0.0.1:5000")
    
    return True

if __name__ == "__main__":
    main()