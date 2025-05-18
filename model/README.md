# Model Directory

This directory stores the trained leprosy detection model in `.h5` format.

## Model Information

- Filename: `leprosy_detection_model.h5`
- Architecture: MobileNetV2 base with custom classification head
- Input Shape: 224x224x3 (RGB image)
- Output: Binary classification (0 - No Leprosy, 1 - Leprosy)

## Training

The model is trained using Kaggle datasets containing leprosy and non-leprosy images. The training process includes:

1. Data preprocessing and augmentation
2. Transfer learning with MobileNetV2 pre-trained on ImageNet
3. Fine-tuning on leprosy-specific data
4. Validation to ensure accuracy and prevent overfitting

## Usage

The model is automatically loaded by the application during startup. If no model file is found, a new model architecture will be created but will require training before making accurate predictions.

To train the model:

1. Ensure Kaggle API credentials are set up
2. Use the `train_model()` function in `model_utils.py`
3. The trained model will be saved to this directory

## Explainable AI

The application uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the regions of the input image that most influenced the model's decision, making the AI more explainable and transparent.
