# Leprosy Detection AI - Quick Start Guide

This guide will help you quickly set up and run the Leprosy Detection AI application on your local machine.

## Prerequisites

- Python 3.8 or higher
- Git (to clone the repository)
- Kaggle credentials (optional, only needed for retraining the model)

## Steps to Run the Application

### 1. Download or Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Set Up the Environment

Create and activate a virtual environment (recommended):

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install email-validator flask flask-login flask-sqlalchemy flask-wtf gunicorn matplotlib numpy opencv-python pandas pillow psycopg2-binary requests scikit-learn sqlalchemy tensorflow trafilatura werkzeug wtforms
```

### 4. Set Up Project Structure

```bash
python setup.py
```

This script will:
- Create all necessary directories
- Check for dependencies
- Set up sample images (if available)
- Prepare the model for use

### 5. Run the Application

```bash
python main.py
```

### 6. Access the Application

Open your web browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

## Testing the Application

### Sample Images

The application includes sample images for testing. To use them:

1. Navigate to [http://127.0.0.1:5000/samples](http://127.0.0.1:5000/samples)
2. Click on any of the sample images to analyze them
3. View the results with the GradCAM visualization

### User Account

To use all features:

1. Register a new account at [http://127.0.0.1:5000/register](http://127.0.0.1:5000/register)
2. Log in with your credentials
3. Go to the Dashboard
4. Upload and analyze your own images
5. View your analysis history

## Features

- **AI-powered detection**: Analyzes skin images to detect signs of leprosy
- **Explainable AI**: Provides GradCAM visualizations to explain model decisions
- **User accounts**: Register, log in, and track your analysis history
- **Test samples**: Pre-loaded samples for immediate testing
- **Doctor recommendations**: Suggests medical professionals for positive cases

## Troubleshooting

- **Images not displaying**: Run `python setup_project_structure.py` to recreate all necessary directories and copy sample images
- **Application not starting**: Ensure all dependencies are installed and Python 3.8+ is being used
- **Model errors**: The pre-trained model is included in the repository. If you encounter issues, you can retrain it (requires Kaggle credentials)

## Advanced: Retraining the Model

To retrain the model with the original datasets (optional):

1. Set up Kaggle credentials:
   ```bash
   # Windows
   set KAGGLE_USERNAME=your_username
   set KAGGLE_KEY=your_api_key
   
   # macOS/Linux
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```

2. Run the preparation and training scripts:
   ```bash
   python prepare_kaggle_data.py
   python prepare_test_samples.py
   python retrain_model.py
   ```