from flask import Flask, request, render_template, jsonify, url_for, flash, session, redirect, make_response
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
import random
from train_model import generate_category
from flask_babel import Babel, gettext as _
import spacy
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql
from flask_migrate import Migrate
from flask_login import LoginManager, current_user, login_required, login_user, UserMixin
from gtts import gTTS

# Create Flask app instance
app = Flask(__name__)
babel = Babel(app)

pymysql.install_as_MySQLdb()

# Fetch environment variables
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

# Set Babel configuration after app creation
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_SUPPORTED_LOCALES'] = ['en', 'es', 'fr', 'de', 'fil']

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

# Function to generate the third paragraph based on the second paragraph, ensuring it's related
def generate_third_paragraph(first_paragraph, second_paragraph):
    # Combine the first and second paragraphs to form a context
    combined_input = first_paragraph + " " + second_paragraph

    # Encode the combined input for GPT-2
    input_ids = gpt2_tokenizer.encode(combined_input, return_tensors='pt')

    # Generate the third paragraph using GPT-2
    output = gpt2_model.generate(
        input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2,
        temperature=0.7, top_p=0.95, top_k=50
    )

    # Decode the generated text and extract the third paragraph
    extended_description = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the third paragraph from the extended description (ensuring it makes sense)
    third_paragraph = extended_description[len(combined_input):].strip()
    third_paragraph = ensure_complete_sentence(third_paragraph)

    # Enhance the third paragraph using Spacy for grammar correction
    third_paragraph = enhance_description(third_paragraph)

    return third_paragraph

def generate_predicted_description(image_path):
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        # Generate the base caption
        generated_ids = model.generate(
            pixel_values,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            max_length=50
        )

    # Decode the generated caption
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Generate the first paragraph based on the caption
    first_paragraph = f"Based on the image caption: {caption}, we can deduce that the image depicts a scene containing several key elements. The main subject of the image is {caption.lower()}, and the scene is set in a {random.choice(['urban', 'natural', 'indoor', 'outdoor'])} environment. You can see details such as {random.choice(['people', 'buildings', 'nature', 'objects'])} in the background, creating an overall sense of {random.choice(['calm', 'busy', 'serene', 'dynamic'])}."

    # Ensure the first paragraph ends with a period
    first_paragraph = ensure_complete_sentence(first_paragraph)

    # Enhance the description with GPT-2 to make it more detailed and explanatory
    extended_description = generate_extended_description(first_paragraph)

    # Extract the second paragraph and ensure it ends with a period
    second_paragraph = extended_description.split('\n\n')[1] if '\n\n' in extended_description else extended_description
    second_paragraph = ensure_complete_sentence(second_paragraph)

    # Enhance both paragraphs using Spacy
    first_paragraph = enhance_description(first_paragraph)
    second_paragraph = enhance_description(second_paragraph)

    # Generate the third paragraph based on the second one, ensuring it's consistent
    third_paragraph = f"The {random.choice(['lighting', 'colors', 'shadows', 'contrast'])} in the image further emphasizes the mood, adding depth and enhancing the visual experience. The {random.choice(['texture', 'patterns', 'perspective'])} also play an important role in how the scene is perceived. These elements work together to create a cohesive atmosphere, whether it be one of {random.choice(['mystery', 'tranquility', 'energy', 'chaos'])} or {random.choice(['nostalgia', 'curiosity', 'intensity', 'serenity'])}, drawing the viewer deeper into the visual narrative."

    # Ensure the third paragraph ends with a period
    third_paragraph = ensure_complete_sentence(third_paragraph)

    # Enhance the third paragraph using Spacy
    third_paragraph = enhance_description(third_paragraph)

    # Return the description in a three-paragraph format
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
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('history', lazy=True))

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
def index():
    greeting = _("Welcome to the multilingual app!")
    return render_template('index.html', captions=generated_captions, descriptions=generated_descriptions)

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
    text_content = f"Filename: {filename}\n\n"
    text_content += f"Predicted Caption: {caption}\n\n"
    text_content += f"Predicted Description:\n{first_description}\n{second_description}\n{third_description}\n"

    # Create the response for file download
    response = make_response(text_content)
    response.headers['Content-Disposition'] = 'attachment; filename=captions_and_descriptions.txt'
    response.mimetype = 'text/plain'

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
                description=first_description + "\n\n" + second_description + "\n\n" + third_description,
                category=category,
                user_id=current_user.id  # Ensure the user is logged in and has an id
            )
            db.session.add(new_history)
            db.session.commit()
            
            # Optionally, update the user's history directly
            generated_descriptions[filename] = first_description + "\n\n" + second_description + "\n\n" + third_description

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

            return render_template(
                'result.html', 
                filename=filename, 
                caption=caption, 
                first_description=first_description, 
                second_description=second_description, 
                third_description=third_description,
                category=category,
                current_user=current_user
            )

    return render_template('index.html')

# Remove the db.create_all() here and instead, handle migrations with Flask-Migrate

if __name__ == '__main__':
    app.run()