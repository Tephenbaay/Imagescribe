import json

# Load the existing JSON data
with open('data/captions_train2017.json', 'r') as f:
    data = json.load(f)

# Write it back to the file with indentation
with open('data/captions_train2017.json', 'w') as f:
    json.dump(data, f, indent=4)  # Adjust the indent value for your preference
