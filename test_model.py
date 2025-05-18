#!/usr/bin/env python3
"""
Test the trained leprosy detection model with a few sample images.
"""

import os
import logging
import matplotlib.pyplot as plt
from model_utils import load_model, preprocess_image, predict_image
from xai_utils import generate_gradcam

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_on_samples():
    """Test the model on sample images from each category"""
    model = load_model('model/leprosy_classifier.pkl')
    
    if not model:
        logger.error("Failed to load model")
        return
    
    logger.info("Model loaded successfully")
    
    # Get metadata
    metadata = model.get_metadata()
    logger.info(f"Model metadata: {metadata}")
    
    # Create output directory for test results
    os.makedirs('test_results', exist_ok=True)
    
    # Test positive sample
    pos_samples_dir = 'static/samples/positive'
    if os.path.exists(pos_samples_dir):
        pos_samples = os.listdir(pos_samples_dir)
        if pos_samples:
            logger.info(f"Testing positive sample: {pos_samples[0]}")
            test_sample(model, os.path.join(pos_samples_dir, pos_samples[0]), expected=True)
    
    # Test negative sample
    neg_samples_dir = 'static/samples/negative'
    if os.path.exists(neg_samples_dir):
        neg_samples = os.listdir(neg_samples_dir)
        if neg_samples:
            logger.info(f"Testing negative sample: {neg_samples[0]}")
            test_sample(model, os.path.join(neg_samples_dir, neg_samples[0]), expected=False)
    
    # Test irrelevant sample (should be classified as negative)
    irr_samples_dir = 'static/samples/irrelevant'
    if os.path.exists(irr_samples_dir):
        irr_samples = os.listdir(irr_samples_dir)
        if irr_samples:
            logger.info(f"Testing irrelevant sample: {irr_samples[0]}")
            test_sample(model, os.path.join(irr_samples_dir, irr_samples[0]), expected=False)
    
    logger.info("Model testing completed")

def test_sample(model, image_path, expected=None):
    """Test model prediction on a single image"""
    try:
        # Preprocess image
        img = preprocess_image(image_path)
        
        # Make prediction
        prediction, confidence = predict_image(model, img)
        
        # Generate GradCAM
        gradcam_path = generate_gradcam(model, image_path, os.path.basename(image_path))
        
        # Log results
        result_str = "Positive" if prediction else "Negative"
        expected_str = "Positive" if expected else "Negative" if expected is not None else "Unknown"
        
        logger.info(f"Image: {image_path}")
        logger.info(f"Prediction: {result_str} with {confidence:.2f} confidence")
        logger.info(f"Expected: {expected_str}")
        
        if expected is not None:
            if prediction == expected:
                logger.info("✓ Correct prediction")
            else:
                logger.info("✗ Incorrect prediction")
        
        logger.info(f"GradCAM visualization: {gradcam_path}")
        logger.info("-" * 50)
        
    except Exception as e:
        logger.error(f"Error testing sample {image_path}: {e}")

if __name__ == "__main__":
    test_model_on_samples()