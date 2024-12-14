from flask import Flask, request, render_template, jsonify, url_for, flash, session, redirect, make_response
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
import random
from train_model import generate_category
import spacy
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql
from flask_migrate import Migrate
from flask_login import LoginManager, current_user, login_required, login_user, UserMixin
from gtts import gTTS
import re
from datetime import datetime

# Create Flask app instance
app = Flask(__name__)

pymysql.install_as_MySQLdb()

 #Fetch environment variables
DB_HOST = os.getenv('AWS_RDS_HOST')
DB_PORT = os.getenv('AWS_RDS_PORT', '3306')  # Default to 5432 for PostgreSQL
DB_USER = os.getenv('AWS_RDS_USER')
DB_PASSWORD = os.getenv('AWS_RDS_PASSWORD')
DB_NAME = os.getenv('AWS_RDS_DB_NAME')

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://admin:Imagescribe11!@imagescribe.cx6aooymq47o.ap-southeast-2.rds.amazonaws.com:3306/imagescribe'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

try:
    connection = pymysql.connect(
        host="imagescribe.cx6aooymq47o.ap-southeast-2.rds.amazonaws.com",
        user="admin",
        password="Imagescribe11!",
        database="imagescribe",
        port=3306
    )
    print("Database connection successful!")
    connection.close()
except Exception as e:
    print(f"Error connecting to database: {e}")


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Replace 'login' with your actual login route

app.secret_key = '9b1e5db5e7f14d2aa8e4ac2f6e3d2e33'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)  # Ensure this exists

    def __repr__(self):
        return f'<User {self.username}>'

    # This is necessary for Flask-Login
    def is_active(self):
        return True  # Return True if the user is active, else False

    def get_id(self):
        return str(self.id)  # Ensure that the ID is returned as a string

# Directory where your images will be stored
image_directory = os.path.join(app.root_path, 'static', 'uploads')

# Ensure the 'uploads' directory exists inside 'static'
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

# Load the trained BLIP model and processor
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Load GPT-2 Model and Tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load SpaCy for grammar and vocabulary correction
nlp = spacy.load("en_core_web_sm")

# Function to generate extended description using GPT-2
def generate_extended_description(caption):
    input_ids = gpt2_tokenizer.encode(caption, return_tensors='pt')
    output = gpt2_model.generate(
        input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2,
        temperature=0.7, top_p=0.95, top_k=50
    )
    extended_description = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return extended_description

# Function to load images
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

# Function to generate captions using the BLIP model
def generate_caption(image_path):
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, temperature=1.0, top_k=50, top_p=0.95)
    
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

# Function to ensure a paragraph ends with a period
def ensure_complete_sentence(paragraph):
    if not paragraph.endswith('.'):
        paragraph = paragraph.rstrip(',.!') + '.'
    return paragraph

# Function to improve grammar and vocabulary using Spacy
def enhance_description(description):
    doc = nlp(description)
    enhanced_text = " ".join([sent.text for sent in doc.sents])  # Fix grammar issues by tokenizing and reconstructing
    return enhanced_text

# Function to ensure a paragraph ends with a period
def ensure_complete_sentence(paragraph):
    if not paragraph.endswith('.'):
        paragraph = paragraph.rstrip(',.!') + '.'
    return paragraph

# Function to generate a predicted description for the uploaded image based on the caption
def generate_predicted_description(image_path):
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            max_length=100
        )

    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    first_paragraph = (
        f"Based on the image caption: {caption}, we can deduce that the image depicts a scene containing several key elements. "
        f"The main subject of the image is {caption.lower()}, and the scene is set in a {random.choice(['urban', 'natural', 'indoor', 'outdoor'])} "
        f"environment. You can see details such as {random.choice(['people', 'buildings', 'nature', 'objects'])} in the background, creating an "
        f"overall sense of {random.choice(['calm', 'busy', 'serene', 'dynamic'])}."
    )
    first_paragraph = ensure_complete_sentence(first_paragraph)

    extended_description = generate_extended_description(first_paragraph)

    # Extract the second paragraph and ensure it ends with a period
    second_paragraph = extended_description.split('\n\n')[1] if '\n\n' in extended_description else extended_description
    second_paragraph = ensure_complete_sentence(second_paragraph)

    first_paragraph = enhance_description(first_paragraph)
    second_paragraph = enhance_description(second_paragraph)

    third_paragraph = (
        f"Together, these descriptions highlight that the image combines elements of {caption.lower()} and its environment to evoke a "
        f"{random.choice(['vivid', 'thought-provoking', 'nostalgic', 'inspiring'])} experience. This underscores the interplay between "
        f"the main subject and its context, offering a comprehensive view that is both engaging and informative."
    )
    third_paragraph = ensure_complete_sentence(third_paragraph)
    third_paragraph = enhance_description(third_paragraph)

    return first_paragraph, second_paragraph, third_paragraph

# Load generated captions/descriptions from the file
def load_generated_data(filepath):
    data = {}
    try:
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.split('|')
                if len(parts) == 2:
                    filename_part = parts[0].strip().split(': ')
                    text_part = parts[1].strip().split(': ')
                    if len(filename_part) == 2 and len(text_part) == 2:
                        filename = filename_part[1]
                        text = text_part[1]
                        data[filename] = text
                    else:
                        print(f"Line format is incorrect in file {filepath}: {line.strip()}")
                else:
                    print(f"Line format is incorrect in file {filepath}: {line.strip()}")
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    return data

# Load generated captions and descriptions when the app starts
generated_captions = load_generated_data('generated_captions.txt')
generated_descriptions = load_generated_data('generated_descriptions.txt')

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    caption = db.Column(db.String(500), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    first_description = db.Column(db.Text, nullable=False)
    second_description = db.Column(db.Text, nullable=False)
    third_description = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('history', lazy=True))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Add this line to store the creation timestamp

    def __repr__(self):
        return f'<History {self.filename}>'

@app.route('/')
def main():
    return render_template('home.html')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))  # Adjust based on your User model

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Query the user by email
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            # Log the user in using Flask-Login
            login_user(user)

            # Flash a success message
            flash("Logged in successfully!", "success")
            
            # Redirect to the page the user was trying to access (or to index by default)
            next_page = request.args.get('next')  # If there was a "next" argument (from a protected page)
            return redirect(next_page or url_for('index'))
        else:
            # If the login failed, flash an error message
            flash("Incorrect email or password.", "error")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if the user already exists in the database
        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            return "User already exists with that email", 400
        
        # Hash the password before storing it
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        # Create a new user
        new_user = User(username=username, email=email, password=hashed_password)
        
        # Add the new user to the session and commit
        db.session.add(new_user)
        db.session.commit()

        # Flash a success message
        flash("Account created successfully! You can now log in.")
        
        # Redirect to login page
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/index')
@login_required
def index():
    # Fetch user history (images uploaded by the logged-in user)
    upload_history = History.query.filter_by(user_id=current_user.id).all()
    
    # Fetch additional results to display in the main content (if needed)
    return render_template('index.html', 
                    captions=generated_captions, 
                    descriptions=generated_descriptions, 
                    upload_history=upload_history,  
                    current_user=current_user)

@app.route("/home", methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")

@app.route("/forget")
def forget():
    return render_template("forget.html")

@app.route("/user")
def user():
    return render_template("user.html")

@app.route('/history')
@login_required  # Ensure the user is logged in to view the history
def history():
    user_history = History.query.filter_by(user_id=current_user.id).all()
    return render_template('history.html', history=user_history)


uploads_directory = os.path.join('static', 'uploads')
os.makedirs(uploads_directory, exist_ok=True)

@app.route('/download_text', methods=['POST'])
def download_text():
    # Retrieve data from the form
    filename = request.form.get('filename')
    caption = request.form.get('caption')
    first_description = request.form.get('first_description')
    second_description = request.form.get('second_description')
    third_description = request.form.get('third_description')

    # Create the text content
    text_content = (
        f"Filename: {filename}\n\n"
        f"Caption: {caption}\n\n"
        f"First Description: {first_description}\n\n"
        f"Second Description: {second_description}\n\n"
        f"Third Description: {third_description}\n\n"
    )

    # Create a response to download the text file
    response = make_response(text_content)
    response.headers["Content-Disposition"] = f"attachment; filename={filename}.txt"
    response.headers["Content-Type"] = "text/plain"
    return response

@app.route('/submit', methods=['POST', 'GET'])
@login_required  # Ensure that the user is logged in
def upload():
    if request.method == 'POST':
        if 'my_image' not in request.files:
            return "No file uploaded.", 400

        file = request.files['my_image']

        # Check the file size (example: limit to 3 MB)
        if file.content_length > 3 * 1024 * 1024:  # 3 MB in bytes
            return render_template("index.html", error="You can only upload a maximum of 3MB per image.", results=[], history=[])

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(uploads_directory, filename)
            file.save(file_path)

            # Generate the caption, description, and category for the uploaded image
            caption = generate_caption(file_path)
            first_description, second_description, third_description = generate_predicted_description(file_path)
            category = generate_category(file_path)
            
            print(f"Generated Caption: {caption}")
            print(f"Generated Description: {first_description} {second_description} {third_description}")
            print(f"Determined Category: {category}")
            
            # Save the image data into the history table
            new_history = History(
                filename=filename,
                caption=caption,
                first_description=first_description,
                second_description=second_description,
                third_description=third_description,
                category=category,
                user_id=current_user.id  # Ensure the user is logged in and has an id
            )
            db.session.add(new_history)
            db.session.commit()

            # Generate audio for caption and description
            caption_audio_path = os.path.join('static', 'audio', f"{file.filename}_caption.mp3")
            tts_caption = gTTS(text=caption, lang='en')
            tts_caption.save(caption_audio_path)

            description_audio_path = os.path.join('static', 'audio', f"{file.filename}_description.mp3")
            tts_description = gTTS(text=first_description, lang='en')
            tts_description.save(description_audio_path)

            description_audio_path = os.path.join('static', 'audio', f"{file.filename}_description.mp3")
            tts_description = gTTS(text=second_description, lang='en')
            tts_description.save(description_audio_path)

            description_audio_path = os.path.join('static', 'audio', f"{file.filename}_description.mp3")
            tts_description = gTTS(text=third_description, lang='en')
            tts_description.save(description_audio_path)

            # Retrieve the upload history for the current user
            upload_history = History.query.filter_by(user_id=current_user.id).order_by(History.created_at.desc()).all()
            upload_history = upload_history[:15]  # Limit to the most recent 15 entries

            return render_template(
                'result.html', 
                filename=filename, 
                caption=caption, 
                first_description=first_description, 
                second_description=second_description,
                third_description=third_description,
                category=category,
                current_user=current_user,
                upload_history=upload_history  # Pass the history to the result.html template
            )

    return render_template('index.html')

@app.route('/history_image/<filename>', methods=['GET'])
@login_required
def history_image(filename):
    # Fetch the image details from the history table based on filename
    history = History.query.filter_by(filename=filename, user_id=current_user.id).first()
    
    if history:
        # Use the description saved in the database

        return render_template(
            'result.html',
            filename=history.filename,
            caption=history.caption,
            first_description=history.first_description,
            second_description=history.second_description,
            third_description=history.third_description,
            category=history.category,
            current_user=current_user
        )
    
    return redirect(url_for('index'))  # If not found, redirect to the home page

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)