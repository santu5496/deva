import os
import logging
import glob
import shutil
import json
from flask import Flask, flash, redirect, render_template, request, url_for, jsonify, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from utils import model_to_dict, to_json, serialize_model, normalize_path, join_paths
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from datetime import datetime
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define database base class
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy
db = SQLAlchemy(model_class=Base)

# Create Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure database with SQLite
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///leprosy_detection.db"
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Initialize database
db.init_app(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Import models and utilities
with app.app_context():
    # Import models
    from models import User, Image, DoctorSuggestion, Result
    
    # Import utilities
    from xai_utils import generate_gradcam, is_skin_image
    from model_utils import load_model, preprocess_image, predict_image
    
    # Create directories if they don't exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs("model", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)
    
    # Create database tables
    db.create_all()
    
    # Load model
    try:
        model = load_model()
        app.logger.info("Model loaded successfully")
    except Exception as e:
        app.logger.error(f"Error loading model: {e}")
        model = None

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Context processor to add variables to all templates
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Utility function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    from forms import RegistrationForm
    
    form = RegistrationForm()
    
    if form.validate_on_submit():
        # Check if user already exists
        existing_user = User.query.filter_by(username=form.username.data).first()
        existing_email = User.query.filter_by(email=form.email.data).first()
        
        if existing_user:
            flash('Username already exists', 'danger')
            return render_template('register.html', form=form)
        
        if existing_email:
            flash('Email already registered', 'danger')
            return render_template('register.html', form=form)
        
        # Create new user
        hashed_password = generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, email=form.email.data, password_hash=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful. Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    from forms import LoginForm
    
    form = LoginForm()
    
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        if user and check_password_hash(user.password_hash, form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Login failed. Please check your username and password', 'danger')
    
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    from forms import UploadForm
    
    form = UploadForm()
    
    if form.validate_on_submit():
        if 'image' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['image']
        
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            new_filename = f"{current_user.id}_{timestamp}_{filename}"
            # Use path utility functions for cross-platform compatibility
            file_path = join_paths('static', 'uploads', new_filename)
            file.save(file_path)
            
            # Process image
            try:
                if not model:
                    flash('Model not loaded. Cannot process image.', 'danger')
                    return redirect(url_for('upload'))
                
                # Check if the image is a skin image
                if not is_skin_image(file_path):
                    flash('The uploaded image does not appear to be a skin image. Please upload a valid skin image.', 'warning')
                    return redirect(url_for('upload'))
                
                # Preprocess image for model
                img_array = preprocess_image(file_path)
                
                # Make prediction
                prediction, confidence = predict_image(model, img_array)
                
                # Generate GradCAM visualization
                gradcam_image_path = generate_gradcam(model, file_path, new_filename)
                
                # Save image and result to database
                image_record = Image(
                    filename=new_filename,
                    path=file_path,
                    user_id=current_user.id
                )
                db.session.add(image_record)
                db.session.flush()  # To get the image_id before commit
                
                result = Result(
                    image_id=image_record.id,
                    prediction=bool(prediction),
                    confidence=float(confidence),
                    gradcam_path=gradcam_image_path,
                    timestamp=datetime.now()
                )
                db.session.add(result)
                db.session.commit()
                
                # Redirect to result page
                return redirect(url_for('result', result_id=result.id))
            
            except Exception as e:
                app.logger.error(f"Error processing image: {e}")
                flash(f'Error processing image: {str(e)}', 'danger')
                return redirect(url_for('upload'))
        
        else:
            flash('Invalid file type. Allowed types: png, jpg, jpeg', 'danger')
            return redirect(request.url)
    
    return render_template('upload.html', form=form)

@app.route('/result/<int:result_id>')
@login_required
def result(result_id):
    result = Result.query.get_or_404(result_id)
    image = Image.query.get_or_404(result.image_id)
    
    # Check if user owns this result
    if image.user_id != current_user.id:
        flash('Unauthorized access', 'danger')
        return redirect(url_for('dashboard'))
    
    # Find doctor suggestions if leprosy detected with high confidence
    doctors = None
    if result.prediction and result.confidence > 0.7:
        doctors = DoctorSuggestion.query.all()
    
    return render_template('result.html', result=result, image=image, doctors=doctors)

@app.route('/history')
@login_required
def history():
    # Get user's image history with results
    user_images = Image.query.filter_by(user_id=current_user.id).all()
    image_ids = [img.id for img in user_images]
    results = Result.query.filter(Result.image_id.in_(image_ids)).order_by(Result.timestamp.desc()).all()
    
    # Convert results and images to dictionaries for safe serialization
    results_list = []
    images_dict = {}
    
    for result in results:
        result_dict = {
            'id': result.id,
            'image_id': result.image_id,
            'prediction': result.prediction,
            'confidence': result.confidence,
            'gradcam_path': result.gradcam_path,
            'timestamp': result.timestamp.strftime('%Y-%m-%d %H:%M:%S') if result.timestamp else None
        }
        results_list.append(result_dict)
    
    for img in user_images:
        images_dict[img.id] = {
            'id': img.id,
            'filename': img.filename,
            'path': img.path,
            'upload_date': img.upload_date.strftime('%Y-%m-%d %H:%M:%S') if img.upload_date else None,
            'user_id': img.user_id
        }
    
    return render_template('history.html', results=results_list, images=images_dict)

@app.route('/doctors')
@login_required
def doctors():
    doctors_list = DoctorSuggestion.query.all()
    return render_template('doctors.html', doctors=doctors_list)

@app.route('/samples')
def samples():
    # Create static/samples directories if they don't exist using path utilities
    samples_dir = join_paths('static', 'samples')
    positive_dir = join_paths(samples_dir, 'positive')
    negative_dir = join_paths(samples_dir, 'negative')
    irrelevant_dir = join_paths(samples_dir, 'irrelevant')
    
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)
    os.makedirs(irrelevant_dir, exist_ok=True)
    
    # Copy sample images from dataset to static folder if needed
    dataset_dir = join_paths('dataset', 'leprosy_dataset')
    
    # Copy positive images if static folder is empty
    if len(os.listdir(positive_dir)) == 0:
        for i, img_path in enumerate(glob.glob(join_paths(dataset_dir, "positive", "*.*"))):
            if i >= 10:  # Limit to 10 images
                break
            dest = join_paths(positive_dir, os.path.basename(img_path))
            shutil.copy(img_path, dest)
    
    # Copy negative images if static folder is empty
    if len(os.listdir(negative_dir)) == 0:
        for i, img_path in enumerate(glob.glob(join_paths(dataset_dir, "negative", "*.*"))):
            if i >= 10:  # Limit to 10 images
                break
            dest = join_paths(negative_dir, os.path.basename(img_path))
            shutil.copy(img_path, dest)
    
    # Copy irrelevant images if static folder is empty
    if len(os.listdir(irrelevant_dir)) == 0:
        for i, img_path in enumerate(glob.glob(join_paths(dataset_dir, "irrelevant", "*.*"))):
            if i >= 10:  # Limit to 10 images
                break
            dest = join_paths(irrelevant_dir, os.path.basename(img_path))
            shutil.copy(img_path, dest)
    
    # Get list of sample images for each category using cross-platform path handling
    positive_images = [f for f in os.listdir(positive_dir) if os.path.isfile(join_paths(positive_dir, f))]
    negative_images = [f for f in os.listdir(negative_dir) if os.path.isfile(join_paths(negative_dir, f))]
    irrelevant_images = [f for f in os.listdir(irrelevant_dir) if os.path.isfile(join_paths(irrelevant_dir, f))]
    
    # Sort images to ensure consistent order
    positive_images.sort()
    negative_images.sort()
    irrelevant_images.sort()
    
    # Limit to 10 images per category
    positive_images = positive_images[:10]
    negative_images = negative_images[:10]
    irrelevant_images = irrelevant_images[:10]
    
    return render_template(
        'samples.html', 
        positive_images=positive_images,
        negative_images=negative_images,
        irrelevant_images=irrelevant_images
    )

@app.route('/upload_sample/<category>/<filename>')
def upload_sample(category, filename):
    """Process a sample image for testing"""
    # Validate the category
    if category not in ['positive', 'negative', 'irrelevant']:
        flash('Invalid category', 'danger')
        return redirect(url_for('samples'))
    
    # Construct the sample image path using path utility functions
    sample_path = join_paths('static', 'samples', category, filename)
    
    if not os.path.exists(sample_path):
        flash('Sample image not found', 'danger')
        return redirect(url_for('samples'))
    
    # Copy the sample to the uploads folder with a timestamp to make it unique
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Handle user authentication - if not logged in, use a guest user ID
    user_id = current_user.id if current_user.is_authenticated else 0
    
    new_filename = f"{user_id}_{timestamp}_{filename}"
    # Use path utility functions for cross-platform compatibility
    upload_path = join_paths('static', 'uploads', new_filename)
    
    try:
        # Copy the sample image to the uploads folder
        shutil.copy(sample_path, upload_path)
        
        # Process the image
        if not model:
            flash('Model not loaded. Cannot process image.', 'danger')
            return redirect(url_for('samples'))
        
        # Preprocess image for model
        img_array = preprocess_image(upload_path)
        
        # Make prediction
        prediction, confidence = predict_image(model, img_array)
        
        # Generate GradCAM visualization
        gradcam_image_path = generate_gradcam(model, upload_path, new_filename)
        
        # Create a temporary result object for display without saving to database
        # if user is not logged in
        if not current_user.is_authenticated:
            temp_result = {
                'id': None,
                'prediction': bool(prediction),
                'confidence': float(confidence),
                'gradcam_path': gradcam_image_path,
                'timestamp': datetime.now()
            }
            
            temp_image = {
                'id': None,
                'filename': new_filename,
                'path': upload_path
            }
            
            # Find doctor suggestions if leprosy detected with high confidence
            doctors = None
            if prediction and confidence > 0.7:
                doctors = DoctorSuggestion.query.all()
                
            return render_template(
                'result.html', 
                result=temp_result, 
                image=temp_image, 
                doctors=doctors,
                is_sample=True
            )
        else:
            # Save image and result to database for logged in users
            image_record = Image(
                filename=new_filename,
                path=upload_path,
                user_id=current_user.id
            )
            db.session.add(image_record)
            db.session.flush()  # To get the image_id before commit
            
            result = Result(
                image_id=image_record.id,
                prediction=bool(prediction),
                confidence=float(confidence),
                gradcam_path=gradcam_image_path,
                timestamp=datetime.now()
            )
            db.session.add(result)
            db.session.commit()
            
            # Redirect to result page
            return redirect(url_for('result', result_id=result.id))
    
    except Exception as e:
        app.logger.error(f"Error processing sample image: {e}")
        flash(f'Error processing image: {str(e)}', 'danger')
        return redirect(url_for('samples'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_code=404, error_message="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
