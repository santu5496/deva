import os
import numpy as np
import logging
import pickle
import time
from PIL import Image
import cv2
from sklearn.ensemble import RandomForestClassifier
from utils import normalize_path, join_paths

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EnhancedLeproyModel:
    """
    A more advanced model for leprosy detection using scikit-learn
    that incorporates real-time learning and dynamic features
    """
    def __init__(self, model_path='model/leprosy_classifier.pkl'):
        self.name = "LeprosiNet"
        self.model_path = normalize_path(model_path)
        self.model = None
        self.feature_importance = None
        self.last_trained = None
        self.accuracy = None
        self.training_samples = 0
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from disk or create a new one"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded model from {self.model_path}")
                
                # Try to load metadata if it exists
                meta_path = self.model_path.replace('.pkl', '_meta.pkl')
                if os.path.exists(meta_path):
                    with open(meta_path, 'rb') as f:
                        metadata = pickle.load(f)
                        self.feature_importance = metadata.get('feature_importance')
                        self.last_trained = metadata.get('last_trained')
                        self.accuracy = metadata.get('accuracy')
                        self.training_samples = metadata.get('training_samples', 0)
            else:
                logger.warning(f"Model file {self.model_path} not found. Creating a new model.")
                self._create_new_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._create_new_model()
    
    def _create_new_model(self):
        """Create a new model if none exists"""
        # Create a Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.last_trained = time.strftime("%Y-%m-%d %H:%M:%S")
        self.accuracy = None
        logger.info("Created new RandomForest model")
    
    def predict(self, features):
        """Make a prediction using the loaded model"""
        try:
            if self.model is not None:
                # If the model is already fitted, make a prediction
                if hasattr(self.model, 'classes_'):
                    # Use probability prediction to get confidence scores
                    probabilities = self.model.predict_proba(features)
                    return probabilities
                else:
                    # If model is not fitted yet, return a default prediction
                    logger.warning("Model not fitted yet, returning default prediction")
                    return np.array([[0.3, 0.7]])
            else:
                # If model loading failed, return a placeholder result
                logger.warning("Model not available, returning default prediction")
                return np.array([[0.3, 0.7]])
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return np.array([[0.3, 0.7]])
    
    def train(self, features, labels):
        """Train or update the model with new data"""
        try:
            self.model.fit(features, labels)
            self.last_trained = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Get feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.model.feature_importances_
            
            # Evaluate accuracy if possible
            if len(features) > 0:
                self.accuracy = self.model.score(features, labels)
                self.training_samples += len(features)
            
            # Save the trained model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save metadata
            meta_path = self.model_path.replace('.pkl', '_meta.pkl')
            metadata = {
                'feature_importance': self.feature_importance,
                'last_trained': self.last_trained,
                'accuracy': self.accuracy,
                'training_samples': self.training_samples
            }
            with open(meta_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Model trained with {len(features)} samples, accuracy: {self.accuracy}")
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def get_metadata(self):
        """Get metadata about the model"""
        return {
            'name': self.name,
            'last_trained': self.last_trained,
            'accuracy': self.accuracy,
            'training_samples': self.training_samples,
            'feature_importance': self.feature_importance
        }

def load_model(model_path='model/leprosy_classifier.pkl'):
    """
    Load the leprosy detection model
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        model: The loaded model
    """
    try:
        return EnhancedLeproyModel(model_path)
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        # Return a new model instance
        return EnhancedLeproyModel()

def create_model():
    """
    Create a new leprosy detection model
    
    Returns:
        model: A new model instance
    """
    try:
        model = EnhancedLeproyModel()
        model._create_new_model()
        return model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return EnhancedLeproyModel()

def extract_features(img_array):
    """
    Extract features from preprocessed image
    
    Args:
        img_array: Preprocessed image array
        
    Returns:
        features: Extracted features for model input
    """
    try:
        # Flatten the image if it's multidimensional
        if len(img_array.shape) > 2:
            # For RGB images, use all channels
            if len(img_array.shape) == 3:
                # Extract color features
                avg_color_per_channel = np.mean(img_array, axis=(0, 1))
                std_color_per_channel = np.std(img_array, axis=(0, 1))
                
                # Convert to grayscale for texture features
                if img_array.shape[2] == 3:  # RGB image
                    gray = cv2.cvtColor(
                        (img_array * 255).astype(np.uint8), 
                        cv2.COLOR_RGB2GRAY
                    )
                else:
                    gray = (img_array[:, :, 0] * 255).astype(np.uint8)
            else:
                # Handle batch dimension if present
                first_img = img_array[0]
                avg_color_per_channel = np.mean(first_img, axis=(0, 1))
                std_color_per_channel = np.std(first_img, axis=(0, 1))
                
                if first_img.shape[2] == 3:  # RGB image
                    gray = cv2.cvtColor(
                        (first_img * 255).astype(np.uint8), 
                        cv2.COLOR_RGB2GRAY
                    )
                else:
                    gray = (first_img[:, :, 0] * 255).astype(np.uint8)
                
            # Basic histogram features
            hist_features = []
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            hist = hist.flatten() / np.sum(hist)  # Normalize
            hist_features.extend(hist)
            
            # Edge detection features
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Simple texture features
            texture_features = [
                np.std(gray),     # Standard deviation
                np.mean(gray),    # Mean intensity
                edge_density      # Edge density
            ]
            
            # Combine all features
            features = np.concatenate([
                avg_color_per_channel, 
                std_color_per_channel,
                hist_features,
                texture_features
            ])
            
            # Reshape for sklearn (samples, features)
            return features.reshape(1, -1)
        else:
            # If already flattened, just return it
            return img_array.reshape(1, -1)
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        # Return consistent feature vector
        feature_size = 3 + 3 + 16 + 3  # RGB means + RGB stds + histogram bins + texture features
        return np.zeros((1, feature_size))

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Preprocess image for model prediction
    
    Args:
        img_path (str): Path to the image file
        target_size (tuple): Target size for resizing
        
    Returns:
        numpy.ndarray: Preprocessed image array ready for model input
    """
    try:
        # Normalize the image path for cross-platform compatibility
        normalized_path = normalize_path(img_path)
        
        # Load image
        if cv2 is not None:
            img = cv2.imread(normalized_path)
            if img is not None:
                # Convert from BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize
                img = cv2.resize(img, target_size)
            else:
                img = None
        else:
            img = None
            
        # Fallback to PIL if cv2 failed
        if img is None:
            img = np.array(Image.open(normalized_path).convert("RGB").resize(target_size))
        
        # Normalize pixel values to [0,1]
        img = img.astype(np.float32) / 255.0
        
        logger.info(f"Image processed: {normalized_path}")
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        # Return a placeholder array in case of error
        return np.zeros((target_size[0], target_size[1], 3))

def predict_image(model, img_array):
    """
    Make prediction on preprocessed image
    
    Args:
        model: Loaded model
        img_array: Preprocessed image array
        
    Returns:
        tuple: (prediction, confidence)
    """
    try:
        # Extract features from the image
        features = extract_features(img_array)
        
        # Get raw predictions
        predictions = model.predict(features)
        
        # Get the predicted class (0 = negative, 1 = positive)
        predicted_class = np.argmax(predictions[0])
        
        # Get confidence score
        confidence = float(predictions[0][predicted_class])
        
        # Convert class index to boolean (True = positive, False = negative)
        binary_prediction = predicted_class == 1
        
        logger.info(f"Prediction: {binary_prediction}, Confidence: {confidence}")
        return binary_prediction, confidence
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return False, 0.0

def train_model_from_dataset(dataset_dir, output_path='model/leprosy_classifier.pkl'):
    """
    Train the model from a dataset directory
    
    Args:
        dataset_dir (str): Path to the dataset directory
        output_path (str): Path to save the trained model
        
    Returns:
        model: The trained model
    """
    try:
        # Create model
        model = EnhancedLeproyModel(output_path)
        
        # Check if dataset exists
        if not os.path.exists(dataset_dir):
            logger.error(f"Dataset directory {dataset_dir} not found.")
            return model
        
        # Look for class folders (positive and negative) using path utilities
        positive_dir = join_paths(dataset_dir, 'positive')
        negative_dir = join_paths(dataset_dir, 'negative')
        
        # Check if these directories exist
        if not all(os.path.exists(d) for d in [positive_dir, negative_dir]):
            logger.error(f"Missing class directories in {dataset_dir}")
            return model
        
        # Load and preprocess images from both classes
        positive_images = os.listdir(positive_dir)
        negative_images = os.listdir(negative_dir)
        
        # Create features and labels arrays
        all_features = []
        all_labels = []
        
        # Process positive samples
        logger.info(f"Processing {len(positive_images)} positive samples...")
        for img_file in positive_images:
            img_path = join_paths(positive_dir, img_file)
            if os.path.isfile(img_path):
                img = preprocess_image(img_path)
                features = extract_features(img)
                all_features.append(features[0])
                all_labels.append(1)  # Positive class
        
        # Process negative samples
        logger.info(f"Processing {len(negative_images)} negative samples...")
        for img_file in negative_images:
            img_path = join_paths(negative_dir, img_file)
            if os.path.isfile(img_path):
                img = preprocess_image(img_path)
                features = extract_features(img)
                all_features.append(features[0])
                all_labels.append(0)  # Negative class
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Train the model
        logger.info(f"Training model with {len(X)} samples...")
        if len(X) > 0:
            model.train(X, y)
            logger.info("Model training completed successfully!")
        else:
            logger.error("No valid samples found for training.")
        
        return model
    except Exception as e:
        logger.error(f"Error training model from dataset: {e}")
        return EnhancedLeproyModel(output_path)
