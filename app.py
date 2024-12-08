from flask import Flask, request, render_template, jsonify, url_for, flash, session, redirect, make_response
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
import random
from train_model import generate_category
import spacy
from werkzeug.security import generate_password_hash, check_password_hash
from gtts import gTTS

# Create Flask app instance
app = Flask(__name__)


# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:imagescribe@localhost/imagescribe'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# db = SQLAlchemy(app)
# migrate = Migrate(app, db)

# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = 'login'  # Replace 'login' with your actual login route

app.secret_key = '9b1e5db5e7f14d2aa8e4ac2f6e3d2e33'

# Set Babel configuration after app creation

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

# Function to generate a predicted description for the uploaded image based on the caption
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

    # Return the description in a two-paragraph format
    return first_paragraph, second_paragraph

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

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    return render_template('signup.html')

@app.route('/')
def index():
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
def history():
    return render_template('history.html')

uploads_directory = os.path.join('static', 'uploads')
os.makedirs(uploads_directory, exist_ok=True)

@app.route('/download_text', methods=['POST'])
def download_text():
    # Retrieve data from the form
    filename = request.form.get('filename')
    caption = request.form.get('caption')
    first_description = request.form.get('first_description')
    second_description = request.form.get('second_description')

    # Create the text content
    text_content = f"Filename: {filename}\n\n"
    text_content += f"Predicted Caption: {caption}\n\n"
    text_content += f"Predicted Description:\n{first_description}\n{second_description}\n"

    # Create the response for file download
    response = make_response(text_content)
    response.headers['Content-Disposition'] = 'attachment; filename=captions_and_descriptions.txt'
    response.mimetype = 'text/plain'

    return response

@app.route('/submit', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        if 'my_image' not in request.files:
            return "No file uploaded.", 400

        file = request.files['my_image']

        # Check the file size (example: limit to 3 MB)
        if file.content_length > 3 * 1024 * 1024:  # 3 MB in bytes
            return render_template("index.html", error="You can only upload a maximum of 3MB per image.", results=[])

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(uploads_directory, filename)
            file.save(file_path)

            # Generate the caption, description, and category for the uploaded image
            caption = generate_caption(file_path)
            first_description, second_description = generate_predicted_description(file_path)
            category = generate_category(file_path)
            
            print(f"Generated Caption: {caption}")
            print(f"Generated Description: {first_description} {second_description}")
            print(f"Determined Category: {category}")
            
            # Save the image data into the history table
            # db.session.add()
            # db.session.commit()
            
            # Optionally, update the user's history directly
            generated_descriptions[filename] = first_description + "\n\n" + second_description

            caption_audio_path = os.path.join('static', 'audio', f"{file.filename}_caption.mp3")
            tts_caption = gTTS(text=caption, lang='en')
            tts_caption.save(caption_audio_path)

            description_audio_path = os.path.join('static', 'audio', f"{file.filename}_description.mp3")
            tts_description = gTTS(text=first_description, lang='en')
            tts_description.save(description_audio_path)

            description_audio_path = os.path.join('static', 'audio', f"{file.filename}_description.mp3")
            tts_description = gTTS(text=second_description, lang='en')
            tts_description.save(description_audio_path)

            return render_template(
                'result.html', 
                filename=filename, 
                caption=caption, 
                first_description=first_description, 
                second_description=second_description, 
                category=category,
            )

    return render_template('index.html')

# Remove the db.create_all() here and instead, handle migrations with Flask-Migrate

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)