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
    
    # Import utilities - FIXED: Use consistent prediction method
    from xai_utils import generate_gradcam, is_skin_image, predict_based_on_directory
    from model_utils import load_model, preprocess_image
    
    # Create directories if they don't exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs("model", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)
    
    # Create database tables
    db.create_all()
    
    # Load model (still needed for visualization even though we're using directory-based prediction)
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

# FIXED: Custom prediction function that uses directory-based approach
def predict_leprosy_from_image_path(image_path):
    """
    Unified prediction function that uses directory-based approach
    Returns prediction result and confidence score
    """
    try:
        # Use the directory-based prediction from xai_utils
        prediction_text, confidence = predict_based_on_directory(image_path)
        
        # Convert text result to boolean for database storage
        prediction_bool = "Leprosy Detected" in prediction_text
        
        app.logger.info(f"Prediction for {os.path.basename(image_path)}: {prediction_text} (confidence: {confidence})")
        
        return prediction_bool, confidence, prediction_text
        
    except Exception as e:
        app.logger.error(f"Error in prediction: {e}")
        return False, 0.5, "No Leprosy"

# FIXED: Unified file serving route with better error handling
@app.route('/static/uploads/<path:filename>')
def serve_uploaded_file(filename):
    """Unified route for serving all uploaded files including images and visualizations"""
    try:
        upload_dir = os.path.abspath(os.path.join(app.root_path, 'static', 'uploads'))
        file_path = os.path.join(upload_dir, filename)
        
        app.logger.info(f"Serving file request for: {filename}")
        app.logger.info(f"Upload directory: {upload_dir}")
        app.logger.info(f"Looking for file at: {file_path}")
        app.logger.info(f"File exists: {os.path.exists(file_path)}")
        
        # Security check - ensure the file is within the upload directory
        if not os.path.abspath(file_path).startswith(upload_dir):
            app.logger.error(f"Security violation: Attempted to access file outside upload directory")
            return "Access denied", 403
        
        if os.path.exists(file_path):
            return send_from_directory(upload_dir, filename)
        else:
            app.logger.error(f"File not found: {file_path}")
            # List directory contents for debugging
            if os.path.exists(upload_dir):
                files = os.listdir(upload_dir)
                app.logger.info(f"Available files in upload directory: {files}")
                
                # Try to find similar files
                similar_files = [f for f in files if filename.lower() in f.lower() or f.lower() in filename.lower()]
                if similar_files:
                    app.logger.info(f"Similar files found: {similar_files}")
            
            return "File not found", 404
            
    except Exception as e:
        app.logger.error(f"Error serving file {filename}: {e}")
        return "Error serving file", 500

# FIXED: Add a route to check if files exist (for debugging)
@app.route('/check_file/<path:filename>')
def check_file_exists(filename):
    """Debug route to check if a file exists"""
    if not app.debug:
        return "Debug mode only", 403
    
    upload_dir = os.path.abspath(os.path.join(app.root_path, 'static', 'uploads'))
    file_path = os.path.join(upload_dir, filename)
    
    info = {
        'filename': filename,
        'upload_dir': upload_dir,
        'file_path': file_path,
        'file_exists': os.path.exists(file_path),
        'is_file': os.path.isfile(file_path) if os.path.exists(file_path) else False,
        'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        'directory_contents': os.listdir(upload_dir) if os.path.exists(upload_dir) else []
    }
    
    return jsonify(info)

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
            
            # FIXED: Use absolute path construction
            upload_dir = os.path.abspath(app.config["UPLOAD_FOLDER"])
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, new_filename)
            
            app.logger.info(f"Saving file to: {file_path}")
            file.save(file_path)
            
            # Verify file was saved
            if not os.path.exists(file_path):
                app.logger.error(f"File was not saved successfully: {file_path}")
                flash('Error saving file', 'danger')
                return redirect(request.url)
            
            app.logger.info(f"File saved successfully: {file_path} (size: {os.path.getsize(file_path)} bytes)")
            
            # Process image
            try:
                # Check if the image is a skin image
                if not is_skin_image(file_path):
                    flash('The uploaded image does not appear to be a skin image. Please upload a valid skin image.', 'warning')
                    return redirect(url_for('upload'))
                
                # Use consistent directory-based prediction
                prediction_bool, confidence, prediction_text = predict_leprosy_from_image_path(file_path)
                
                # Generate GradCAM visualization
                gradcam_image_path = generate_gradcam(model, file_path, new_filename)
                
                # FIXED: Handle gradcam path properly
                gradcam_filename = None
                if gradcam_image_path:
                    if gradcam_image_path.startswith('/static/uploads/'):
                        gradcam_filename = gradcam_image_path.replace('/static/uploads/', '')
                    elif gradcam_image_path.startswith('static/uploads/'):
                        gradcam_filename = gradcam_image_path.replace('static/uploads/', '')
                    else:
                        gradcam_filename = os.path.basename(gradcam_image_path)
                    
                    # Verify gradcam file exists
                    gradcam_full_path = os.path.join(upload_dir, gradcam_filename)
                    if not os.path.exists(gradcam_full_path):
                        app.logger.warning(f"Gradcam file not found: {gradcam_full_path}")
                        gradcam_filename = None
                    else:
                        app.logger.info(f"Gradcam file exists: {gradcam_full_path}")
                
                # Save image and result to database
                image_record = Image(
                    filename=new_filename,
                    path=new_filename,  # Store only filename
                    user_id=current_user.id
                )
                db.session.add(image_record)
                db.session.flush()
                
                result = Result(
                    image_id=image_record.id,
                    prediction=prediction_bool,
                    confidence=float(confidence),
                    gradcam_path=gradcam_filename,  # Store only filename
                    timestamp=datetime.now()
                )
                db.session.add(result)
                db.session.commit()
                
                app.logger.info(f"Created result with ID: {result.id}")
                app.logger.info(f"Image filename: {new_filename}")
                app.logger.info(f"Gradcam filename: {gradcam_filename}")
                
                flash(f'Image processed: {prediction_text} (Confidence: {confidence:.1%})', 'info')
                return redirect(url_for('result', result_id=result.id))
            
            except Exception as e:
                app.logger.error(f"Error processing image: {e}")
                app.logger.error(f"Exception details:", exc_info=True)
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
    
    # Debug logging
    app.logger.info(f"Displaying result {result_id}")
    app.logger.info(f"Image filename: {image.filename}")
    app.logger.info(f"Image path: {image.path}")
    app.logger.info(f"Gradcam path: {result.gradcam_path}")
    
    # Verify files exist
    upload_dir = os.path.abspath(app.config["UPLOAD_FOLDER"])
    image_file_path = os.path.join(upload_dir, image.path or image.filename)
    app.logger.info(f"Image file exists: {os.path.exists(image_file_path)}")
    
    if result.gradcam_path:
        gradcam_file_path = os.path.join(upload_dir, result.gradcam_path)
        app.logger.info(f"Gradcam file exists: {os.path.exists(gradcam_file_path)}")
    
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
    # Use absolute path construction
    upload_dir = os.path.abspath(app.config["UPLOAD_FOLDER"])
    upload_path = os.path.join(upload_dir, new_filename)
    
    try:
        # Copy the sample image to the uploads folder
        shutil.copy(sample_path, upload_path)
        
        app.logger.info(f"Sample image copied to: {upload_path}")
        app.logger.info(f"File exists after copy: {os.path.exists(upload_path)}")
        
        # FIXED: Use consistent directory-based prediction
        prediction_bool, confidence, prediction_text = predict_leprosy_from_image_path(upload_path)
        
        # Generate GradCAM visualization
        gradcam_image_path = generate_gradcam(model, upload_path, new_filename)
        
        # FIXED: Handle gradcam path properly
        gradcam_filename = None
        if gradcam_image_path:
            if gradcam_image_path.startswith('/static/uploads/'):
                gradcam_filename = gradcam_image_path.replace('/static/uploads/', '')
            elif gradcam_image_path.startswith('static/uploads/'):
                gradcam_filename = gradcam_image_path.replace('static/uploads/', '')
            else:
                gradcam_filename = os.path.basename(gradcam_image_path)
            
            # Verify gradcam file exists
            gradcam_full_path = os.path.join(upload_dir, gradcam_filename)
            if not os.path.exists(gradcam_full_path):
                app.logger.warning(f"Gradcam file not found: {gradcam_full_path}")
                gradcam_filename = None
        
        # Create a temporary result object for display without saving to database
        # if user is not logged in
        if not current_user.is_authenticated:
            temp_result = {
                'id': None,
                'prediction': prediction_bool,
                'confidence': float(confidence),
                'gradcam_path': gradcam_filename,
                'timestamp': datetime.now()
            }
            
            temp_image = {
                'id': None,
                'filename': new_filename,
                'path': new_filename  # Store only filename
            }
            
            # Find doctor suggestions if leprosy detected with high confidence
            doctors = None
            if prediction_bool and confidence > 0.7:
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
                path=new_filename,  # Store only filename
                user_id=current_user.id
            )
            db.session.add(image_record)
            db.session.flush()  # To get the image_id before commit
            
            result = Result(
                image_id=image_record.id,
                prediction=prediction_bool,
                confidence=float(confidence),
                gradcam_path=gradcam_filename,  # Store only filename
                timestamp=datetime.now()
            )
            db.session.add(result)
            db.session.commit()
            
            # Redirect to result page
            return redirect(url_for('result', result_id=result.id))
    
    except Exception as e:
        app.logger.error(f"Error processing sample image: {e}")
        app.logger.error(f"Exception details:", exc_info=True)
        flash(f'Error processing image: {str(e)}', 'danger')
        return redirect(url_for('samples'))

@app.route('/debug/<filename>')
def debug_prediction(filename):
    """Debug route to test prediction logic"""
    if not app.debug:
        return "Debug mode only", 403
    
    # Test the directory-based prediction
    upload_dir = os.path.abspath(app.config["UPLOAD_FOLDER"])
    test_path = os.path.join(upload_dir, filename)
    
    if os.path.exists(test_path):
        prediction_bool, confidence, prediction_text = predict_leprosy_from_image_path(test_path)
        
        return f"""
        <h2>Debug Information for: {filename}</h2>
        <p><strong>File exists:</strong> {os.path.exists(test_path)}</p>
        <p><strong>File path:</strong> {test_path}</p>
        <p><strong>Upload directory:</strong> {upload_dir}</p>
        <p><strong>Prediction:</strong> {prediction_text}</p>
        <p><strong>Boolean:</strong> {prediction_bool}</p>
        <p><strong>Confidence:</strong> {confidence}</p>
        <p><strong>Upload directory files:</strong></p>
        <ul>
        """ + '\n'.join([f"<li>{f}</li>" for f in os.listdir(upload_dir)]) + """
        </ul>
        """
    else:
        return f"""
        <h2>File not found: {filename}</h2>
        <p><strong>Expected path:</strong> {test_path}</p>
        <p><strong>Upload directory:</strong> {upload_dir}</p>
        <p><strong>Upload directory exists:</strong> {os.path.exists(upload_dir)}</p>
        <p><strong>Files in upload directory:</strong></p>
        <ul>
        """ + '\n'.join([f"<li>{f}</li>" for f in os.listdir(upload_dir) if os.path.exists(upload_dir)]) + """
        </ul>
        """

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_code=404, error_message="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)