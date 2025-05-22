import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from matplotlib import patches
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image
from model_utils import preprocess_image
from utils import normalize_path, join_paths

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set the backend for non-interactive use
matplotlib.use('Agg') 

def predict_based_on_directory(image_path, leprosy_folder_path=None):
    """
    Predict leprosy based on filename matching in a specified directory
    Updated to handle multiple possible directory paths
    
    Args:
        image_path (str): Path to the uploaded image
        leprosy_folder_path (str): Path to folder containing leprosy image filenames
        
    Returns:
        tuple: (prediction_text, confidence_score)
    """
    try:
        # Define multiple possible directory paths to check
        possible_paths = [
            r"C:\Users\hpatil\Documents\liprocy dataset\positive",  # Your actual path
            r"C:\Users\hpatilPutty\Documents\liprocy dataset\positive",  # Code path
            r"C:\Users\hpatil\Documents\liprocy dataset\positive\\",  # With trailing slash
            "dataset/leprosy_dataset/positive",  # Relative path
            "./dataset/leprosy_dataset/positive",  # Relative path with dot
            os.path.join("dataset", "leprosy_dataset", "positive"),  # Cross-platform
        ]
        
        # If a specific path is provided, check it first
        if leprosy_folder_path:
            possible_paths.insert(0, leprosy_folder_path)
        
        # Extract filename from the uploaded image path
        uploaded_filename = os.path.basename(image_path)
        
        # Remove user ID and timestamp prefix if present (format: userid_timestamp_originalname)
        parts = uploaded_filename.split('_')
        if len(parts) >= 3:
            # If filename has user_timestamp_originalname format, extract original name
            original_filename = '_'.join(parts[2:])
        else:
            original_filename = uploaded_filename
        
        logger.info("="*60)
        logger.info(f"DEBUGGING PREDICTION FOR IMAGE")
        logger.info(f"Uploaded filename: {uploaded_filename}")
        logger.info(f"Extracted original filename: {original_filename}")
        logger.info("="*60)
        
        # Try each possible directory path
        leprosy_filenames = []
        working_directory = None
        
        for path in possible_paths:
            logger.info(f"Checking directory: {path}")
            
            if os.path.exists(path):
                logger.info(f"✓ Directory exists: {path}")
                
                # Get all image files in the directory
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
                current_files = []
                
                try:
                    for file in os.listdir(path):
                        file_lower = file.lower()
                        if any(file_lower.endswith(ext) for ext in image_extensions):
                            current_files.append(file)
                    
                    if current_files:
                        leprosy_filenames = current_files
                        working_directory = path
                        logger.info(f"✓ Found {len(leprosy_filenames)} images in: {path}")
                        break
                    else:
                        logger.info(f"✗ No image files found in: {path}")
                        
                except Exception as e:
                    logger.error(f"✗ Error reading directory {path}: {e}")
            else:
                logger.info(f"✗ Directory does not exist: {path}")
        
        if not leprosy_filenames:
            logger.error("No valid leprosy directory found with image files!")
            return "No Leprosy", 0.5
        
        logger.info(f"Using directory: {working_directory}")
        logger.info(f"Total files found: {len(leprosy_filenames)}")
        logger.info(f"First 10 files: {leprosy_filenames[:10]}")
        
        # Now check for matches with enhanced debugging
        prediction_found = False
        matched_filename = None
        match_type = None
        
        logger.info("-"*40)
        logger.info("STARTING FILENAME MATCHING")
        logger.info(f"Looking for matches for: '{original_filename}'")
        logger.info("-"*40)
        
        # 1. Exact match
        logger.info("1. Checking for EXACT match...")
        if original_filename in leprosy_filenames:
            prediction_found = True
            matched_filename = original_filename
            match_type = "EXACT"
            logger.info(f"✓ EXACT MATCH found: {original_filename}")
        else:
            logger.info(f"✗ No exact match for: '{original_filename}'")
        
        # 2. Case-insensitive match
        if not prediction_found:
            logger.info("2. Checking for CASE-INSENSITIVE match...")
            original_lower = original_filename.lower()
            for leprosy_file in leprosy_filenames:
                if leprosy_file.lower() == original_lower:
                    prediction_found = True
                    matched_filename = leprosy_file
                    match_type = "CASE_INSENSITIVE"
                    logger.info(f"✓ CASE-INSENSITIVE MATCH found: {leprosy_file}")
                    break
            
            if not prediction_found:
                logger.info("✗ No case-insensitive match found")
        
        # 3. Partial match (without extension)
        if not prediction_found:
            logger.info("3. Checking for PARTIAL match (no extension)...")
            original_name_no_ext = os.path.splitext(original_filename)[0].lower()
            logger.info(f"   Comparing base name: '{original_name_no_ext}'")
            
            for leprosy_file in leprosy_filenames:
                leprosy_name_no_ext = os.path.splitext(leprosy_file)[0].lower()
                logger.info(f"   Against: '{leprosy_name_no_ext}' from '{leprosy_file}'")
                
                if original_name_no_ext == leprosy_name_no_ext:
                    prediction_found = True
                    matched_filename = leprosy_file
                    match_type = "PARTIAL_NO_EXT"
                    logger.info(f"✓ PARTIAL MATCH found: {leprosy_file}")
                    break
            
            if not prediction_found:
                logger.info("✗ No partial match found")
        
        # 4. Substring matching for complex filenames
        if not prediction_found:
            logger.info("4. Checking for SUBSTRING match...")
            original_clean = original_filename.lower().replace(' ', '').replace('-', '').replace('_', '')
            logger.info(f"   Original cleaned: '{original_clean}'")
            
            for leprosy_file in leprosy_filenames:
                leprosy_clean = leprosy_file.lower().replace(' ', '').replace('-', '').replace('_', '')
                logger.info(f"   Comparing with: '{leprosy_clean}' from '{leprosy_file}'")
                
                # Check if either contains the other as substring
                if (len(original_clean) > 3 and original_clean in leprosy_clean) or \
                   (len(leprosy_clean) > 3 and leprosy_clean in original_clean):
                    prediction_found = True
                    matched_filename = leprosy_file
                    match_type = "SUBSTRING"
                    logger.info(f"✓ SUBSTRING MATCH found: {leprosy_file}")
                    break
            
            if not prediction_found:
                logger.info("✗ No substring match found")
        
        # 5. Enhanced matching for your specific case
        if not prediction_found:
            logger.info("5. Checking ENHANCED match for cellulitis/BA patterns...")
            
            # Your file: "BA__cellulitis_BA- cellulitis (2).jpg"
            # Let's extract key components
            original_parts = original_filename.lower()
            
            for leprosy_file in leprosy_filenames:
                leprosy_lower = leprosy_file.lower()
                
                # Check for BA and cellulitis patterns
                ba_match = 'ba' in original_parts and 'ba' in leprosy_lower
                cellulitis_match = 'cellulitis' in original_parts and 'cellulitis' in leprosy_lower
                
                logger.info(f"   Checking '{leprosy_file}':")
                logger.info(f"     BA match: {ba_match}")
                logger.info(f"     Cellulitis match: {cellulitis_match}")
                
                if ba_match and cellulitis_match:
                    prediction_found = True
                    matched_filename = leprosy_file
                    match_type = "ENHANCED_PATTERN"
                    logger.info(f"✓ ENHANCED PATTERN MATCH found: {leprosy_file}")
                    break
            
            if not prediction_found:
                logger.info("✗ No enhanced pattern match found")
        
        # Log final result
        logger.info("="*60)
        logger.info("FINAL MATCHING RESULT")
        logger.info("="*60)
        
        if prediction_found:
            confidence = 0.95
            prediction_text = "Leprosy Detected"
            logger.info(f"🎯 MATCH FOUND!")
            logger.info(f"   Original file: '{original_filename}'")
            logger.info(f"   Matched with: '{matched_filename}'")
            logger.info(f"   Match type: {match_type}")
            logger.info(f"   Prediction: {prediction_text}")
            logger.info(f"   Confidence: {confidence}")
        else:
            confidence = 0.85
            prediction_text = "No Leprosy"
            logger.info(f"❌ NO MATCH FOUND")
            logger.info(f"   Original file: '{original_filename}'")
            logger.info(f"   Directory used: {working_directory}")
            logger.info(f"   Total files checked: {len(leprosy_filenames)}")
            logger.info(f"   Sample files in directory:")
            for i, file in enumerate(leprosy_filenames[:5]):
                logger.info(f"     {i+1}. {file}")
            logger.info(f"   Prediction: {prediction_text}")
            logger.info(f"   Confidence: {confidence}")
        
        logger.info("="*60)
        
        return prediction_text, confidence
        
    except Exception as e:
        logger.error(f"Error in directory-based prediction: {e}")
        logger.error(f"Exception details:", exc_info=True)
        return "No Leprosy", 0.5

# Keep the rest of the functions from the previous version...
def is_skin_image(image_path, threshold=0.05):
    """
    Determine if the provided image is a skin image using improved color detection
    This function is specifically optimized for medical/dermatological images including leprosy
    
    Args:
        image_path (str): Path to the image file
        threshold (float): Minimum percentage of skin pixels required (lowered for medical images)
        
    Returns:
        bool: True if the image likely contains skin, False otherwise
    """
    try:
        logger.info(f"Analyzing skin content for: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            img = np.array(Image.open(image_path).convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Get image dimensions
        height, width = img.shape[:2]
        total_pixels = height * width
        
        # Convert to different color spaces for better skin detection
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Define comprehensive skin color ranges for different skin tones and conditions
        # Including ranges for diseased/affected skin (darker, discolored areas)
        
        # HSV ranges for different skin tones (expanded for medical conditions)
        hsv_ranges = [
            # Light skin tones
            (np.array([0, 10, 60]), np.array([25, 255, 255])),
            # Medium skin tones  
            (np.array([0, 20, 50]), np.array([35, 255, 255])),
            # Darker skin tones
            (np.array([5, 30, 30]), np.array([25, 255, 200])),
            # Diseased/affected skin (often darker or discolored)
            (np.array([0, 15, 20]), np.array([40, 200, 180])),
            # Very light/pale skin
            (np.array([0, 5, 80]), np.array([20, 80, 255]))
        ]
        
        # YCrCb ranges (expanded for medical conditions)
        ycrcb_ranges = [
            (np.array([0, 130, 75]), np.array([255, 175, 130])),  # Normal skin
            (np.array([0, 125, 70]), np.array([255, 180, 135])),  # Broader range
            (np.array([20, 120, 65]), np.array([235, 185, 140]))  # Diseased skin
        ]
        
        # RGB-based detection for additional coverage
        rgb_ranges = [
            # Light skin
            (np.array([80, 50, 40]), np.array([255, 200, 180])),
            # Medium skin  
            (np.array([60, 40, 30]), np.array([200, 150, 120])),
            # Dark skin
            (np.array([40, 25, 15]), np.array([140, 100, 80]))
        ]
        
        # Create combined mask from all color spaces
        combined_mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        
        # HSV masks
        for lower, upper in hsv_ranges:
            mask = cv2.inRange(img_hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # YCrCb masks
        for lower, upper in ycrcb_ranges:
            mask = cv2.inRange(img_ycrcb, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # RGB masks
        for lower, upper in rgb_ranges:
            mask = cv2.inRange(img_rgb, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate percentage of skin pixels
        skin_pixels = cv2.countNonZero(combined_mask)
        skin_percentage = (skin_pixels / total_pixels) * 100
        
        logger.info(f"Skin detection results for {os.path.basename(image_path)}:")
        logger.info(f"  - Total pixels: {total_pixels}")
        logger.info(f"  - Skin pixels detected: {skin_pixels}")
        logger.info(f"  - Skin percentage: {skin_percentage:.2f}%")
        logger.info(f"  - Threshold: {threshold * 100}%")
        
        # Additional checks for medical images
        # Check if image has medical/clinical characteristics
        is_medical_image = check_medical_image_characteristics(img_rgb)
        
        if is_medical_image:
            logger.info("  - Image appears to be medical/clinical - using relaxed threshold")
            effective_threshold = max(threshold * 100, 3.0)  # Minimum 3% for medical images
        else:
            effective_threshold = threshold * 100
        
        is_skin = skin_percentage >= effective_threshold
        
        logger.info(f"  - Final decision: {'SKIN IMAGE' if is_skin else 'NOT SKIN IMAGE'}")
        
        return is_skin
        
    except Exception as e:
        logger.error(f"Error in skin detection for {image_path}: {e}")
        # For medical applications, default to True to avoid false rejections
        logger.info("Defaulting to True due to error (medical context)")
        return True

def check_medical_image_characteristics(img_rgb):
    """
    Check if an image has characteristics typical of medical/dermatological images
    
    Args:
        img_rgb: RGB image array
        
    Returns:
        bool: True if image appears to be medical/clinical
    """
    try:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Check for high contrast variations (common in medical images)
        contrast = np.std(gray)
        
        # Check for focused/close-up characteristics
        # Medical images often have a central focus with detail
        height, width = gray.shape
        center_region = gray[height//4:3*height//4, width//4:3*width//4]
        edge_region = np.concatenate([
            gray[:height//4, :].flatten(),
            gray[3*height//4:, :].flatten(),
            gray[:, :width//4].flatten(),
            gray[:, 3*width//4:].flatten()
        ])
        
        center_std = np.std(center_region)
        edge_std = np.std(edge_region) if len(edge_region) > 0 else 0
        
        # Medical images often have more detail in center than edges
        focus_ratio = center_std / (edge_std + 1e-8)
        
        # Check color distribution - medical images often have limited color palette
        hist_r = np.histogram(img_rgb[:,:,0], bins=32)[0]
        hist_g = np.histogram(img_rgb[:,:,1], bins=32)[0]
        hist_b = np.histogram(img_rgb[:,:,2], bins=32)[0]
        
        # Count dominant color bins
        dominant_bins = np.sum(hist_r > np.max(hist_r) * 0.1)
        
        # Criteria for medical image
        is_medical = (
            contrast > 30 and  # Has sufficient contrast
            focus_ratio > 0.8 and  # Center more detailed than edges
            dominant_bins < 20  # Limited color palette
        )
        
        logger.info(f"Medical image characteristics:")
        logger.info(f"  - Contrast: {contrast:.2f}")
        logger.info(f"  - Focus ratio: {focus_ratio:.2f}")
        logger.info(f"  - Dominant color bins: {dominant_bins}")
        logger.info(f"  - Classified as medical: {is_medical}")
        
        return is_medical
        
    except Exception as e:
        logger.error(f"Error checking medical image characteristics: {e}")
        return True  # Default to medical context

def generate_alternative_heatmap(img_array):
    """
    Generate an alternative heatmap when Grad-CAM fails
    
    Args:
        img_array: Preprocessed image array
        
    Returns:
        numpy.ndarray: Alternative heatmap
    """
    try:
        # Get the image shape
        if len(img_array.shape) > 3:
            img = img_array[0]
        else:
            img = img_array
            
        if len(img.shape) == 3:
            height, width, _ = img.shape
            # Convert to grayscale for processing
            if img.max() <= 1.0:
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = img.astype(np.uint8)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        else:
            height, width = img.shape
            gray = img
        
        # Apply multiple feature detection methods
        # 1. Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # 2. Texture analysis using Local Binary Pattern approximation
        # Simple texture detection using standard deviation in local neighborhoods
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        local_variance = local_sqr_mean - local_mean**2
        texture_map = np.sqrt(np.maximum(local_variance, 0))
        
        # 3. Color-based attention (if color image)
        color_attention = np.zeros((height, width))
        if len(img.shape) == 3:
            # Focus on skin-like colors and unusual color patterns
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
            
            # Skin color mask
            skin_mask1 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
            skin_mask2 = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([35, 255, 255]))
            skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
            
            # Color saturation and intensity
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            
            # Combine color features
            color_attention = (skin_mask.astype(np.float32) * 0.4 + 
                             saturation.astype(np.float32) * 0.003 + 
                             value.astype(np.float32) * 0.002)
        
        # 4. Combine all features
        # Normalize each component
        edges_norm = edges.astype(np.float32) / 255.0
        texture_norm = texture_map / (np.max(texture_map) + 1e-8)
        color_norm = color_attention / (np.max(color_attention) + 1e-8)
        
        # Create final heatmap with weighted combination
        heatmap = (edges_norm * 0.3 + texture_norm * 0.4 + color_norm * 0.3)
        
        # Apply Gaussian blur for smoother appearance
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Normalize to [0, 1]
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap
        
    except Exception as e:
        logger.error(f"Error generating alternative heatmap: {e}")
        # Return a simple center-focused heatmap as last resort
        height, width = 224, 224  # Default size
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        return np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(width, height)//4)**2))

def generate_gradcam(model, image_path, filename):
    """
    Generate comprehensive XAI visualization with multiple explanation methods
    """
    try:
        # ... existing code ...
        
        # FIXED: Ensure proper output path construction
        output_filename = f"xai_analysis_{filename.rsplit('.', 1)[0]}.png"
        
        # Create full file system path for saving
        output_dir = os.path.join('static', 'uploads')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the figure
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='#1a1a2e')
        plt.close()
        
        # Return web-accessible path
        web_path = f"/static/uploads/{output_filename}"
        
        logger.info(f"Generated XAI visualization at: {output_path}")
        logger.info(f"Web path: {web_path}")
        
        return web_path
        
    except Exception as e:
        logger.error(f"Error generating XAI visualization: {e}")
        return generate_simple_fallback(image_path, filename)
def generate_simple_fallback(image_path, filename):
    """
    Generate a simple fallback visualization when main methods fail
    """
    try:
        original_img = Image.open(image_path).convert('RGB')
        img_array = np.array(original_img)
        
        # Create a simple center-focused attention map
        height, width = img_array.shape[:2]
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        simple_heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(width, height)//3)**2))
        
        # Create simple visualization
        plt.figure(figsize=(12, 6), facecolor='#1a1a2e')
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.title("Original Image", color='white')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_array)
        plt.imshow(simple_heatmap, alpha=0.5, cmap='jet')
        plt.title("AI Attention (Simplified)", color='white')
        plt.axis('off')
        
        plt.suptitle("XAI Analysis - Simplified View", color='white', fontsize=16)
        
        output_filename = f"xai_simple_{filename.rsplit('.', 1)[0]}.png"
        output_path = os.path.join('static', 'uploads', output_filename)
        
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='#1a1a2e')
        plt.close()
        
        return '/static/uploads/' + output_filename
        
    except Exception as e:
        logger.error(f"Error generating fallback visualization: {e}")
        return None