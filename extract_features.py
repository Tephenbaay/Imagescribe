import os
import pickle
import numpy as np
import json
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm  # For progress bar

# Load the Xception model without the top layer and with average pooling
model = Xception(include_top=False, pooling='avg')

# Directory where your COCO images are stored
# Update this path to where your COCO images are located
directory = r'C:\Users\Admin\Documents\image-caption-generator-main\data\train2017'  # Update this path

# Load existing features if available, or initialize an empty dictionary
if os.path.exists('features.pkl'):
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
else:
    features = {}

# Load the captions from the COCO dataset
with open('data/captions_train2017.json', 'r') as f:
    train_data = json.load(f)

# Create a set of image IDs from the training captions
image_ids = set(str(ann['image_id']) for ann in train_data['annotations'])

# Loop through the image files in the directory
for filename in tqdm(os.listdir(directory), desc="Extracting Features"):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Fixed the typo here
        # Extract the image ID from the filename
        image_id = filename.split('.')[0]
        
        # Only process if the image ID (str) is in the COCO dataset
        if image_id in image_ids:
            if image_id not in features:  # Only process if the image's features aren't already extracted
                # Construct full image path
                img_path = os.path.join(directory, filename)

                # Load and preprocess the image
                image = load_img(img_path, target_size=(299, 299))  # Resize the image
                image = img_to_array(image)  # Convert image to array
                image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
                image = preprocess_input(image)  # Preprocess the image using Xception's preprocessing

                # Extract features using the model
                feature = model.predict(image)

                # Save the extracted features in the dictionary
                features[image_id] = feature

# Save the updated features to the pickle file
with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)

print(f"Features extracted and saved to features.pkl")
print(f"Total features extracted: {len(features)}")
