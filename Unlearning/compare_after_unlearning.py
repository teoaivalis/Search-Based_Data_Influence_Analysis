import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50


# Load pre-trained VGG16 model
model = ResNet50(weights='imagenet', include_top=False)

def extract_features(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()  # Flatten feature tensor to 1D array

def calculate_cosine_similarity(reference_features, folder_path, model):
    similarity_scores = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            features = extract_features(image_path, model)
            cos_sim = cosine_similarity([reference_features], [features])[0][0]
            similarity_scores.append(cos_sim)
    return similarity_scores

def process_folders(reference_folder, before_folder, after_folder, model):
    data = []

    for subfolder in os.listdir(before_folder):
        reference_image_path = os.path.join(reference_folder, subfolder, '0.png')
        folder_path_1 = os.path.join(before_folder, subfolder)
        folder_path_2 = os.path.join(after_folder, subfolder)

        if os.path.exists(reference_image_path) and os.path.exists(folder_path_1) and os.path.exists(folder_path_2):
            reference_features = extract_features(reference_image_path, model)
            similarity_scores_1 = calculate_cosine_similarity(reference_features, folder_path_1, model)
            similarity_scores_2 = calculate_cosine_similarity(reference_features, folder_path_2, model)

            mean_similarity_1 = np.mean(similarity_scores_1)
            mean_similarity_2 = np.mean(similarity_scores_2)
            min_similarity_1 = np.min(similarity_scores_1)
            min_similarity_2 = np.min(similarity_scores_2)
            max_similarity_1 = np.max(similarity_scores_1)
            max_similarity_2 = np.max(similarity_scores_2)

            data.append([subfolder,
                         mean_similarity_1, min_similarity_1, max_similarity_1
                         mean_similarity_2, min_similarity_2, max_similarity_2])

    return data

# Paths to the parent folders
reference_folder = './outputs_reference_1'
before_folder = './outputs_before_1'
after_folder = './outputs_after_1'

# Process each subfolder and get data
data = process_folders(reference_folder, before_folder, after_folder, model)

# Write data to CSV file
csv_file = './similarity_results.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Folder', 'Mean_1', 'Min_1', 'Max_1', 'Mean_2', 'Min_2', 'Max_2'])
    writer.writerows(data)

print("CSV file has been saved successfully.")

