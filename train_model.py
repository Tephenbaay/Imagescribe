import os
import json
import torch
from PIL import Image
import spacy
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

# Load spaCy model for grammar corrections and key entity extraction
nlp = spacy.load("en_core_web_sm")

# Initialize BLIP model and processor
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Load Sentence-BERT for similarity checking
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize GPT-2 for description generation
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define the maximum length of a caption sequence
max_length_val = 32

# Define categories and associated keywords
categories = {
    "animals": ["dog", "cat", "bird", "elephant", "giraffe", "zebra", "lion", "tiger", "bear", "fish", "shark", "whale", "ducks"],
    "vehicles": ["car", "truck", "bike", "bus", "plane", "train", "boat", "ship", "scooter", "helicopter"],
    "people": ["man", "woman", "child", "girl", "boy", "person", "crowd", "family", "friends"],
    "buildings": ["house", "building", "bridge", "tower", "skyscraper", "castle", "monument", "temple", "church", "mosque"],
    "landscapes": ["mountain", "river", "forest", "desert", "beach", "waterfall", "valley", "lake", "hill", "sunset"],
    "food": ["pizza", "burger", "salad", "fruit", "cake", "ice cream", "sushi", "pasta", "bread", "chocolate"],
    "technology": ["phone", "laptop", "camera", "robot", "tablet", "drone", "computer", "headphones", "microwave", "television"],
    "sports": ["soccer", "basketball", "tennis", "cricket", "golf", "swimming", "cycling", "boxing", "running", "skiing"],
    "nature": ["flower", "tree", "plant", "grass", "sky", "cloud", "rain", "snow", "sun", "moon"],
    "abstract": ["pattern", "shape", "texture", "color", "design", "art", "geometry", "shadow", "reflection"],
    "others": ["miscellaneous", "unknown", "uncategorized"]
}

def generate_category(image_path):
    # Generate caption for the image
    caption = generate_caption(image_path)

    # Match caption keywords to categories
    for category, keywords in categories.items():
        if any(keyword in caption.lower() for keyword in keywords):
            return category

    return "others"  # Default category if no match is found

# Function to categorize images based on captions
def categorize_image(captions):
    for caption in captions:
        for category, keywords in categories.items():
            if any(keyword in caption.lower() for keyword in keywords):
                return category
    return "others"  # Default category if no match

# Function to load images
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

# Function to categorize images based on captions
def categorize_image(captions):
    for caption in captions:
        for category, keywords in categories.items():
            if any(keyword in caption for keyword in keywords):
                return category
    return "others"  # Default category if no match

# Load descriptions from the COCO dataset
def load_descriptions(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    descriptions = {}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        if image_id not in descriptions:
            descriptions[image_id] = []
        descriptions[image_id].append(caption)
    
    return descriptions

# Load the already processed images from the captions file
def load_processed_images(filepath):
    processed_images = set()
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            for line in file:
                if line.strip():  # Ignore empty lines
                    filename = line.split('|')[0].split(': ')[1].strip()
                    processed_images.add(filename)
    return processed_images

# Function to generate captions with added randomness and diversity
def generate_caption(image_path):
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    # Generate the caption with randomness
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, temperature=1.0, top_k=50, top_p=0.95)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return caption

# Function to generate a more refined and elaborate description
def generate_description(image_path):
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        # Generate a detailed caption using BLIP
        generated_ids = model.generate(
            pixel_values,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            max_length=100  # Generate the initial caption
        )
    
    # Decode the generated caption
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Now, let's generate a more detailed description based on this caption
    description = generate_detailed_description(image_path, caption)
    
    return description

# Function to generate a more detailed, multi-paragraph description
def generate_detailed_description(image_path, caption):
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        # Generate a detailed caption using BLIP
        generated_ids = model.generate(
            pixel_values,
            temperature=1.5,  # Increase creativity
            top_k=50,
            top_p=0.95,
            num_beams=8,  # More coherent and fluent output
            max_length=350,  # Ensure longer, more detailed output
            early_stopping=True  # Stop once the model finishes generating
        )
    
    detailed_description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Post-process to ensure the description is elaborate
    detailed_description = f"Based on the image of {caption}, the scene captures a moment where {detailed_description}"

    # Now break it into two paragraphs (with additional logic for more coherence)
    detailed_description = detailed_description.replace(". ", ".\n\n", 1)  # Force a split after the first sentence

    # Ensure that the description is naturally divided into two paragraphs
    if '.' in detailed_description:
        # If there are multiple periods, make sure they split well into two paragraphs
        detailed_description = detailed_description.replace(". ", ".\n\n", 2)  # Add a second paragraph after another period

    return detailed_description

def generate_caption_and_description(image_path):
    # Generate the caption for the uploaded image
    caption = generate_caption(image_path)
    
    # Generate descriptions based on the generated caption (same logic from your `train_model.py`)
    # Assuming you have a list of existing captions for similar images, you can process the description
    # Here we are using a placeholder description logic
    descriptions = load_descriptions('data/captions_train2017.json')  # Assuming this exists
    image_captions = descriptions.get(int(image_path.split('.')[0]), [])
    description = rephrase_and_expand_captions(image_captions, caption)
    
    return caption, description

# Function to filter and rephrase captions into a cohesive description
def rephrase_and_expand_captions(captions, generated_caption):
    """Rephrase the generated caption and expand it into a detailed, two-paragraph description."""
    filtered_captions = filter_relevant_captions(generated_caption, captions)
    input_text = (
        "The following are captions describing an image: "
        f"{'; '.join(filtered_captions)}. Please expand them into a detailed, grammatically correct description. "
        "The first paragraph should provide a general overview of the image, including its main subjects. "
        "The second paragraph should provide more specific details, such as the background, context, or emotions evoked."
    )
    
    inputs = gpt_tokenizer.encode(input_text, return_tensors="pt")
    outputs = gpt_model.generate(
        inputs,
        max_length=300,  # Allow a longer output
        temperature=0.7,  # Moderate creativity
        top_p=0.9,
        num_beams=5,
        early_stopping=True
    )
    
    # Get the generated output as text
    expanded_description = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return expanded_description.strip()

# Updated train_model function with relevance filtering
# Function to process each image and save the generated caption and expanded description
def train_model():
    image_directory = r'C:\Users\Admin\Documents\image-caption-generator-main\data\train2017'
    descriptions = load_descriptions('data/captions_train2017.json')
    print(f"Loaded descriptions with {len(descriptions)} descriptions.")
    
    processed_images = load_processed_images('generated_captions.txt')
    
    with open('generated_captions.txt', 'a') as caption_file, open('generated_descriptions.txt', 'a') as description_file:
        for filename in os.listdir(image_directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                if filename in processed_images:
                    print(f"Skipping already processed image: {filename}")
                    continue
                
                image_id = int(filename.split('.')[0])
                if image_id in descriptions:
                    # Generate the caption
                    caption = generate_caption(os.path.join(image_directory, filename))
                    
                    # Filter and rephrase descriptions, now with expansion
                    image_captions = descriptions[image_id][:5]
                    description = rephrase_and_expand_captions(image_captions, caption)
                    
                    # Categorize and save
                    category = categorize_image([caption])
                    caption_file.write(f"Image: {filename} | Generated Caption: {caption}\n")
                    description_file.write(f"Image: {filename} | Generated Description: {description}\n")
                    
                    category_folder = os.path.join(r'C:\image-caption-generator-main\categories', category)
                    os.makedirs(category_folder, exist_ok=True)
                    save_path = os.path.join(category_folder, filename)
                    os.rename(os.path.join(image_directory, filename), save_path)
                    
                    print(f"Image: {filename} | Category: {category} | Caption: {caption}")
                    print(f"Generated Description: {description}")
                else:
                    print(f"No description found for {image_id}")

# Main execution
if __name__ == '__main__':
    train_model()
