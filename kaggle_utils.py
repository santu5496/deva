import os
import json
import zipfile
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def setup_kaggle_credentials(kaggle_username=None, kaggle_key=None):
    """
    Set up Kaggle API credentials either from parameters or environment variables
    
    Args:
        kaggle_username (str, optional): Kaggle username
        kaggle_key (str, optional): Kaggle API key
        
    Returns:
        bool: True if credentials are set up successfully, False otherwise
    """
    try:
        # Check if credentials are already set
        if os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
            logger.info("Kaggle credentials already exist.")
            return True
            
        # Get credentials from environment variables if not provided
        if not kaggle_username:
            kaggle_username = os.environ.get('KAGGLE_USERNAME')
        if not kaggle_key:
            kaggle_key = os.environ.get('KAGGLE_KEY')
            
        if not kaggle_username or not kaggle_key:
            logger.error("Kaggle credentials not provided and not found in environment variables.")
            return False
            
        # Create .kaggle directory if it doesn't exist
        os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
        
        # Write credentials to kaggle.json
        with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
            json.dump({
                'username': kaggle_username,
                'key': kaggle_key
            }, f)
            
        # Set permissions
        os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
        
        logger.info("Kaggle credentials set up successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up Kaggle credentials: {e}")
        return False

def download_dataset(dataset_name, dataset_dir="dataset"):
    """
    Download a dataset from Kaggle
    
    Args:
        dataset_name (str): Name of the dataset in format 'owner/dataset-name'
        dataset_dir (str): Directory to save the dataset
        
    Returns:
        str: Path to the downloaded dataset directory
    """
    try:
        # Set up Kaggle API
        if not setup_kaggle_credentials():
            return None
            
        api = KaggleApi()
        api.authenticate()
        
        # Create dataset directory if it doesn't exist
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Download dataset
        logger.info(f"Downloading dataset {dataset_name}...")
        api.dataset_download_files(dataset_name, path=dataset_dir, unzip=True)
        
        logger.info(f"Dataset {dataset_name} downloaded successfully.")
        return dataset_dir
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return None

def get_leprosy_datasets():
    """
    Search for and download leprosy-related datasets from Kaggle
    
    Returns:
        list: List of downloaded dataset paths
    """
    try:
        # Set up Kaggle API
        if not setup_kaggle_credentials():
            return []
            
        api = KaggleApi()
        api.authenticate()
        
        # Search for leprosy datasets
        logger.info("Searching for leprosy datasets...")
        datasets = api.dataset_list(search='leprosy')
        
        downloaded_paths = []
        for dataset in datasets:
            dataset_name = f"{dataset.owner}/{dataset.ref}"
            logger.info(f"Found dataset: {dataset_name}")
            
            # Download the dataset
            path = download_dataset(dataset_name)
            if path:
                downloaded_paths.append(path)
                
        return downloaded_paths
        
    except Exception as e:
        logger.error(f"Error searching for leprosy datasets: {e}")
        return []
