import pickle

# Load the existing features from features.pkl
def load_features(features_path):
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    return features

# Save the updated features to features.pkl
def save_features(features_path, features):
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)

# Remove the specified image ID from features
def remove_feature(image_id, features):
    if image_id in features:
        del features[image_id]
        print(f"Removed feature for image ID: {image_id}")
    else:
        print(f"Image ID {image_id} not found in features.")

# Main function
def main():
    features_path = 'features.pkl'  # Path to your features file
    image_id_to_remove = '2258277193_586949ec62'  # Image ID to remove

    # Load features
    features = load_features(features_path)
    print(f"Loaded features with {len(features)} entries.")

    # Remove the specified image ID
    remove_feature(image_id_to_remove, features)

    # Save the updated features back to the file
    save_features(features_path, features)
    print(f"Updated features saved. Total features now: {len(features)}.")

if __name__ == '__main__':
    main()
