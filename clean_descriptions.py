# Define the unwanted phrases or sentences to remove
unwanted_phrases = [
    "A car that seems to be parked illegally behind a legally parked car."
]

def clean_descriptions(file_path):
    # Read all lines from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Clean each line by removing unwanted phrases
    cleaned_lines = []
    for line in lines:
        for phrase in unwanted_phrases:
            line = line.replace(phrase, '').strip()  # Remove unwanted phrases
        cleaned_lines.append(line.strip())

    # Write the cleaned lines back to the file
    with open(file_path, 'w') as file:
        for line in cleaned_lines:
            if line:  # Avoid writing empty lines
                file.write(line + '\n')

# Path to the generated descriptions file
file_path = 'generated_descriptions.txt'

# Clean the file
clean_descriptions(file_path)

print("Descriptions file cleaned successfully!")