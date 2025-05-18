#!/usr/bin/env python3
"""
This script prepares the dataset directory structure for training the leprosy detection model.
It downloads sample images if none are available.
"""

import os
import logging
import argparse
import shutil
from pathlib import Path
import requests
import urllib.request
from PIL import Image
from io import BytesIO
from kaggle_utils import setup_kaggle_credentials, download_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample images for testing if no real dataset is available
SAMPLE_POSITIVE_URLS = [
    "https://www.researchgate.net/profile/Natalia-Fonseca/publication/331555566/figure/fig3/AS:735882270466054@1552439704222/Clinical-aspects-of-lepromatous-leprosy-a-Multiple-infiltrated-patches-and-papules-b.png",
    "https://www.researchgate.net/profile/Salvatore-Ruberto/publication/325119357/figure/fig1/AS:626788084166658@1526431536369/Lepromatous-leprosy-Fig-Courtesy-Dr-Roberto-Antonello-Biella-A-patient-with-leonine.png",
    "https://www.the-hospitalist.org/sites/default/files/styles/medium/public/images/P22_Leprosy.png",
]

SAMPLE_NEGATIVE_URLS = [
    "https://dermnetnz.org/assets/Uploads/normal/normal-skin-1.jpg",
    "https://www.researchgate.net/profile/Mahesh-Mathur/publication/278027498/figure/fig3/AS:294477547548676@1447221145605/Normal-human-skin-H-E-stain-X100.png",
    "https://www.researchgate.net/profile/Muhammad-Sharif-42/publication/335456726/figure/fig3/AS:804785221914625@1568981269026/Sample-images-from-normal-category-of-HAM10000-dataset-17.png",
]

def download_image(url, save_path):
    """Download an image from URL and save it to the specified path"""
    try:
        # Try to download the image
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Save the image
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        logger.info(f"Downloaded image from {url} to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return False

def setup_data_directory(data_dir="dataset/leprosy_dataset", add_samples=True):
    """
    Set up the data directory structure for a binary classification problem:
    - data_dir/positive
    - data_dir/negative
    
    Args:
        data_dir (str): Base directory for the dataset
        add_samples (bool): Whether to add sample images if directories are empty
        
    Returns:
        str: Path to the prepared dataset directory
    """
    # Create directory structure
    positive_dir = os.path.join(data_dir, "positive")
    negative_dir = os.path.join(data_dir, "negative")
    
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)
    
    logger.info(f"Created dataset directories at {data_dir}")
    
    # Check if directories are empty
    if add_samples and (len(os.listdir(positive_dir)) == 0 or len(os.listdir(negative_dir)) == 0):
        logger.info("Directories are empty, adding sample images...")
        
        # Download sample positive images
        for i, url in enumerate(SAMPLE_POSITIVE_URLS):
            download_image(url, os.path.join(positive_dir, f"sample_positive_{i+1}.png"))
        
        # Download sample negative images
        for i, url in enumerate(SAMPLE_NEGATIVE_URLS):
            download_image(url, os.path.join(negative_dir, f"sample_negative_{i+1}.png"))
    
    return data_dir

def download_kaggle_dataset():
    """
    Download a leprosy dataset from Kaggle
    
    Returns:
        str: Path to the downloaded dataset
    """
    # Set up Kaggle credentials
    if not setup_kaggle_credentials():
        logger.error("Failed to set up Kaggle credentials")
        return None
    
    # Try to download a leprosy dataset
    try:
        # First try a specific leprosy dataset
        dataset_path = download_dataset("nikhilroxtomar/leprosy-detection-using-deep-learning", "dataset/leprosy_dataset")
        
        # If not available, try another one
        if not dataset_path:
            dataset_path = download_dataset("aryashah2k/skin-disease-dataset", "dataset/skin_disease_dataset")
        
        # If not available, try another one
        if not dataset_path:
            dataset_path = download_dataset("shubhamgoel27/dermnet", "dataset/dermnet")
        
        # If one of the datasets was downloaded, return the path
        if dataset_path:
            logger.info(f"Downloaded dataset to {dataset_path}")
            return dataset_path
        else:
            logger.warning("No suitable dataset found on Kaggle")
            return None
    
    except Exception as e:
        logger.error(f"Error downloading dataset from Kaggle: {e}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare dataset for leprosy detection')
    parser.add_argument('--data-dir', type=str, default='dataset/leprosy_dataset', 
                        help='Directory for the dataset')
    parser.add_argument('--kaggle', action='store_true', 
                        help='Download dataset from Kaggle')
    parser.add_argument('--no-samples', action='store_true',
                        help='Do not add sample images if directories are empty')
    args = parser.parse_args()
    
    # Try to download dataset from Kaggle if requested
    if args.kaggle:
        dataset_path = download_kaggle_dataset()
        if dataset_path:
            # If successful, set data_dir to the downloaded dataset path
            args.data_dir = dataset_path
    
    # Set up data directory structure
    dataset_path = setup_data_directory(args.data_dir, not args.no_samples)
    
    # Print summary
    logger.info(f"Dataset preparation complete.")
    logger.info(f"  Dataset directory: {dataset_path}")
    logger.info(f"  Positive examples: {len(os.listdir(os.path.join(dataset_path, 'positive')))}")
    logger.info(f"  Negative examples: {len(os.listdir(os.path.join(dataset_path, 'negative')))}")

if __name__ == "__main__":
    main()