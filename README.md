# Leprosy Detection AI Application

An AI-powered Flask web application for Leprosy detection using advanced machine learning and explainable AI techniques.

## Features

- Deep learning-based image classification for leprosy detection
- Explainable AI (XAI) visualization with GradCAM heatmaps
- User authentication system
- Doctor recommendation integration
- Responsive web interface
- Test samples page for easy testing

## Dataset

This project uses real medical datasets from Kaggle:
- Skin disease dataset for leprosy positive samples: "subirbiswas19/skin-disease-dataset"
- Skin types dataset for healthy skin (negative samples): "shakyadissanayake/oily-dry-and-normal-skin-types-dataset"

## Requirements

```
email-validator
flask
flask-login
flask-sqlalchemy
flask-wtf
gunicorn
kaggle
kagglehub
keras
matplotlib
numpy
openai
opencv-python
opencv-python-headless
pandas
pillow
psycopg2-binary
requests
scikit-learn
sqlalchemy
tensorflow
trafilatura
werkzeug
wtforms
```

## Project Structure

- `app.py`: Main Flask application
- `forms.py`: Flask-WTF form definitions
- `models.py`: SQLAlchemy database models
- `model_utils.py`: Model prediction and training utilities
- `xai_utils.py`: Explainable AI visualization functions
- `prepare_kaggle_data.py`: Data preparation from Kaggle
- `train_model.py`: Model training script
- `test_model_performance.py`: Test evaluation script
- `retrain_model.py`: Script to retrain the model with correct feature dimensions
- `templates/`: HTML templates
- `static/`: Static files (CSS, JS, images)

## Setup Instructions

1. Clone the repository
2. Install the required packages:
   ```
   pip install email-validator flask flask-login flask-sqlalchemy flask-wtf gunicorn matplotlib numpy opencv-python pandas pillow psycopg2-binary requests scikit-learn sqlalchemy tensorflow trafilatura werkzeug wtforms
   ```

3. Set up the project structure:
   ```
   python setup_project_structure.py
   ```
   This script creates all necessary directories and copies sample images.

4. Run the application:
   ```
   python main.py
   ```
   Or with gunicorn:
   ```
   gunicorn --bind 127.0.0.1:5000 main:app
   ```

5. Access the application at [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Optional: Download Datasets and Retrain Model

If you want to download the original datasets and retrain the model:

1. Set up environment variables:
   - KAGGLE_USERNAME: Your Kaggle username
   - KAGGLE_KEY: Your Kaggle API key

2. Run the data preparation script:
   ```
   python prepare_kaggle_data.py
   ```

3. Create test samples:
   ```
   python prepare_test_samples.py
   ```

4. Train the model:
   ```
   python retrain_model.py
   ```

Note: The pre-trained model is already included in the repository, so these steps are optional.

## Web Interface

- `/`: Home page
- `/login`: Login page
- `/register`: User registration
- `/dashboard`: User dashboard
- `/upload`: Upload images for analysis
- `/result/<id>`: View analysis results
- `/history`: View history of analyses
- `/doctors`: View doctor recommendations
- `/samples`: Test with sample images

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source.