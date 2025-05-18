#!/usr/bin/env python3
"""
Test script to evaluate the performance of the leprosy detection model on a set of
positive and negative images, with XAI visualization and result storage in the database.
"""

import os
import csv
import random
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from model_utils import load_model, preprocess_image, predict_image
from xai_utils import generate_gradcam, is_skin_image
from app import db, app, Image, Result

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NUM_SAMPLES_PER_CLASS = 15
POSITIVE_CONFIDENCE_THRESHOLD = 0.90
NEGATIVE_CONFIDENCE_THRESHOLD = 0.95
TEST_RESULTS_DIR = 'test_results'
TEST_SUMMARY_FILE = os.path.join(TEST_RESULTS_DIR, 'summary.csv')

def setup_test_environment():
    """Set up the test environment"""
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    logger.info(f"Test results will be stored in {TEST_RESULTS_DIR}")

def get_test_images(positive_count=NUM_SAMPLES_PER_CLASS, negative_count=NUM_SAMPLES_PER_CLASS):
    """
    Get test images from the test_samples directory
    
    Args:
        positive_count: Number of positive samples to select
        negative_count: Number of negative samples to select
        
    Returns:
        positive_images: List of paths to positive images
        negative_images: List of paths to negative images
    """
    # Positive images from test samples
    positive_dir = 'test_samples/positive'
    if not os.path.exists(positive_dir):
        logger.error(f"Positive test directory {positive_dir} not found")
        logger.warning("Please run prepare_test_samples.py first")
        return [], []
    
    positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir)
                     if os.path.isfile(os.path.join(positive_dir, f))]
    
    # Negative images from test samples
    negative_dir = 'test_samples/negative'
    if not os.path.exists(negative_dir):
        logger.error(f"Negative test directory {negative_dir} not found")
        logger.warning("Please run prepare_test_samples.py first")
        return [], []
    
    negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir)
                     if os.path.isfile(os.path.join(negative_dir, f))]
    
    # Use all available test samples
    positive_selected = positive_files[:positive_count]
    negative_selected = negative_files[:negative_count]
    
    logger.info(f"Selected {len(positive_selected)} positive and {len(negative_selected)} negative test images")
    
    return positive_selected, negative_selected

def evaluate_image(model, img_path, expected_class, user_id=1):
    """
    Evaluate a single image and store results
    
    Args:
        model: The loaded model
        img_path: Path to the image
        expected_class: Expected class (True for positive, False for negative)
        user_id: User ID for storing in database
        
    Returns:
        dict: Results including prediction, confidence, accuracy
    """
    try:
        # Check if image is valid
        if not os.path.exists(img_path):
            logger.error(f"Image not found: {img_path}")
            return None
        
        # Check if it's a skin image
        if not is_skin_image(img_path):
            logger.warning(f"Image may not be a skin image: {img_path}")
        
        # Preprocess image
        img = preprocess_image(img_path)
        
        # Make prediction
        prediction, confidence = predict_image(model, img)
        
        # Generate GradCAM visualization
        filename = os.path.basename(img_path)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        gradcam_filename = f"gradcam_{timestamp}_{filename}"
        gradcam_path = generate_gradcam(model, img_path, gradcam_filename)
        
        # Check if prediction matches expected class
        correct = prediction == expected_class
        expected_str = "Positive" if expected_class else "Negative"
        prediction_str = "Positive" if prediction else "Negative"
        
        # Store result in database with app context
        with app.app_context():
            # Create new image record
            image = Image(
                filename=filename,
                path=img_path,
                upload_date=datetime.now(),
                user_id=user_id
            )
            db.session.add(image)
            db.session.flush()  # Get the image ID without committing
            
            # Create result record
            result = Result(
                image_id=image.id,
                prediction=prediction,
                confidence=float(confidence),
                gradcam_path=gradcam_path if gradcam_path else None,
                timestamp=datetime.now()
            )
            db.session.add(result)
            db.session.commit()
        
        # Log results
        logger.info(f"Image: {filename}")
        logger.info(f"Expected: {expected_str}, Predicted: {prediction_str}, Confidence: {confidence:.4f}")
        logger.info(f"Correct: {correct}")
        if gradcam_path:
            logger.info(f"GradCAM visualization: {gradcam_path}")
        
        return {
            'filename': filename,
            'expected': expected_class,
            'predicted': prediction,
            'confidence': confidence,
            'correct': correct,
            'gradcam_path': gradcam_path
        }
    
    except Exception as e:
        logger.error(f"Error evaluating image {img_path}: {e}")
        return None

def run_performance_test():
    """Run the full performance test on positive and negative images"""
    # Set up test environment
    setup_test_environment()
    
    # Load model
    model = load_model('model/leprosy_classifier.pkl')
    if not model:
        logger.error("Failed to load model. Test aborted.")
        return
    
    logger.info("Model loaded successfully")
    
    # Get test images
    positive_images, negative_images = get_test_images()
    
    # Evaluate positive images
    positive_results = []
    logger.info("Evaluating positive images...")
    for img_path in positive_images:
        result = evaluate_image(model, img_path, True)
        if result:
            positive_results.append(result)
    
    # Evaluate negative images
    negative_results = []
    logger.info("Evaluating negative images...")
    for img_path in negative_images:
        result = evaluate_image(model, img_path, False)
        if result:
            negative_results.append(result)
    
    # Analyze results
    analyze_results(positive_results, negative_results)

def analyze_results(positive_results, negative_results):
    """
    Analyze and summarize test results
    
    Args:
        positive_results: List of result dictionaries for positive images
        negative_results: List of result dictionaries for negative images
    """
    # Calculate metrics
    pos_correct = sum(r['correct'] for r in positive_results)
    neg_correct = sum(r['correct'] for r in negative_results)
    
    pos_total = len(positive_results)
    neg_total = len(negative_results)
    
    pos_accuracy = pos_correct / pos_total if pos_total > 0 else 0
    neg_accuracy = neg_correct / neg_total if neg_total > 0 else 0
    
    total_correct = pos_correct + neg_correct
    total_images = pos_total + neg_total
    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    
    # Calculate average confidence
    pos_avg_confidence = np.mean([r['confidence'] for r in positive_results]) if positive_results else 0
    neg_avg_confidence = np.mean([r['confidence'] for r in negative_results]) if negative_results else 0
    
    # Check for high confidence thresholds
    pos_high_confidence = sum(r['confidence'] >= POSITIVE_CONFIDENCE_THRESHOLD for r in positive_results)
    neg_high_confidence = sum(r['confidence'] >= NEGATIVE_CONFIDENCE_THRESHOLD for r in negative_results)
    
    # Log summary statistics
    logger.info("\n--- PERFORMANCE TEST RESULTS ---")
    logger.info(f"Total images tested: {total_images}")
    logger.info(f"Overall accuracy: {overall_accuracy:.4f}")
    logger.info("\nPositive class:")
    logger.info(f"  Accuracy: {pos_accuracy:.4f} ({pos_correct}/{pos_total})")
    logger.info(f"  Average confidence: {pos_avg_confidence:.4f}")
    logger.info(f"  High confidence (>={POSITIVE_CONFIDENCE_THRESHOLD}): {pos_high_confidence}/{pos_total}")
    logger.info("\nNegative class:")
    logger.info(f"  Accuracy: {neg_accuracy:.4f} ({neg_correct}/{neg_total})")
    logger.info(f"  Average confidence: {neg_avg_confidence:.4f}")
    logger.info(f"  High confidence (>={NEGATIVE_CONFIDENCE_THRESHOLD}): {neg_high_confidence}/{neg_total}")
    
    # Write results to CSV
    with open(TEST_SUMMARY_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Category', 'Metric', 'Value'])
        writer.writerow(['Overall', 'Total Images', total_images])
        writer.writerow(['Overall', 'Accuracy', f"{overall_accuracy:.4f}"])
        writer.writerow(['Positive', 'Images Tested', pos_total])
        writer.writerow(['Positive', 'Correct Predictions', pos_correct])
        writer.writerow(['Positive', 'Accuracy', f"{pos_accuracy:.4f}"])
        writer.writerow(['Positive', 'Average Confidence', f"{pos_avg_confidence:.4f}"])
        writer.writerow(['Positive', f'High Confidence (>={POSITIVE_CONFIDENCE_THRESHOLD})', f"{pos_high_confidence}/{pos_total}"])
        writer.writerow(['Negative', 'Images Tested', neg_total])
        writer.writerow(['Negative', 'Correct Predictions', neg_correct])
        writer.writerow(['Negative', 'Accuracy', f"{neg_accuracy:.4f}"])
        writer.writerow(['Negative', 'Average Confidence', f"{neg_avg_confidence:.4f}"])
        writer.writerow(['Negative', f'High Confidence (>={NEGATIVE_CONFIDENCE_THRESHOLD})', f"{neg_high_confidence}/{neg_total}"])
    
    logger.info(f"Results saved to {TEST_SUMMARY_FILE}")
    
    # Create detailed results CSV
    detailed_csv = os.path.join(TEST_RESULTS_DIR, 'detailed_results.csv')
    with open(detailed_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Expected', 'Predicted', 'Confidence', 'Correct', 'GradCAM Path'])
        
        for result in positive_results:
            writer.writerow([
                result['filename'],
                'Positive',
                'Positive' if result['predicted'] else 'Negative',
                f"{result['confidence']:.4f}",
                result['correct'],
                result['gradcam_path'] if result['gradcam_path'] else ''
            ])
        
        for result in negative_results:
            writer.writerow([
                result['filename'],
                'Negative',
                'Positive' if result['predicted'] else 'Negative',
                f"{result['confidence']:.4f}",
                result['correct'],
                result['gradcam_path'] if result['gradcam_path'] else ''
            ])
    
    logger.info(f"Detailed results saved to {detailed_csv}")
    
    # Plot results
    plot_results(positive_results, negative_results)

def plot_results(positive_results, negative_results):
    """
    Create visualizations of the test results
    
    Args:
        positive_results: List of result dictionaries for positive images
        negative_results: List of result dictionaries for negative images
    """
    # Create confusion matrix
    tp = sum(r['predicted'] == True for r in positive_results)
    fp = sum(r['predicted'] == True for r in negative_results)
    tn = sum(r['predicted'] == False for r in negative_results)
    fn = sum(r['predicted'] == False for r in positive_results)
    
    cm = np.array([[tn, fp], [fn, tp]])
    
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
    cm_path = os.path.join(TEST_RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    # Confidence distribution
    plt.figure(figsize=(12, 6))
    
    # Positive class confidence
    pos_conf = [r['confidence'] for r in positive_results]
    neg_conf = [r['confidence'] for r in negative_results]
    
    plt.hist([pos_conf, neg_conf], bins=10, 
             label=['Positive class', 'Negative class'],
             alpha=0.7, color=['#ff9999', '#66b3ff'])
    
    plt.axvline(x=POSITIVE_CONFIDENCE_THRESHOLD, color='red', linestyle='--', 
                label=f'Positive threshold ({POSITIVE_CONFIDENCE_THRESHOLD})')
    plt.axvline(x=NEGATIVE_CONFIDENCE_THRESHOLD, color='blue', linestyle='--', 
                label=f'Negative threshold ({NEGATIVE_CONFIDENCE_THRESHOLD})')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Confidence Scores')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save plot
    conf_path = os.path.join(TEST_RESULTS_DIR, 'confidence_distribution.png')
    plt.savefig(conf_path)
    plt.close()
    logger.info(f"Confidence distribution saved to {conf_path}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run the performance test
    run_performance_test()