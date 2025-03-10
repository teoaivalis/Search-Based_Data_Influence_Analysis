from keras.preprocessing import image
from numpy import asarray
from numpy import savetxt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
import os
from keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error 
import csv
import tensorflow as tf

# Specify the GPU ID you want to use
gpu_id = '1'

# Set GPU device
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# Check if the specified GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False)

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))

def get_mean_size(input_folder):
    sizes = []

    for filename in os.listdir(input_folder):
        if is_image_file(filename):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                sizes.append((img.shape[1], img.shape[0]))  # Width, Height

    if sizes:
        mean_size = np.mean(sizes, axis=0)
        return mean_size.astype(int)
    else:
        return None

def preprocess_and_extract_embedding(img_path, model):
        try:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
            embeddings = model.predict(img_array)
            flattened_embeddings = embeddings.flatten()
            return flattened_embeddings
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None

def calculate_similarity(embedding1, embedding2, similarity_type='cosine'):
    if embedding1 is not None and embedding2 is not None:
        if similarity_type == 'cosine':
            return cosine_similarity([embedding1], [embedding2])[0][0]
        elif similarity_type == 'mse':
            return mean_squared_error(embedding1, embedding2)
        elif similarity_type == 'kl':
            epsilon = 1e-10  # Small value to avoid division by zero and logarithm of zero
            embedding2 = np.clip(embedding2, epsilon, None)  # Clip values to avoid zero and negative values
            kl_divergence = np.sum(embedding1 * np.log(np.where(embedding1 != 0, embedding1 / embedding2, 1)))
            return kl_divergence
        elif similarity_type == 'manhattan':
            # Compute Manhattan distance
            return np.sum(np.abs(embedding1 - embedding2))
        elif similarity_type == 'minkowski':
            # Compute Minkowski distance (you can specify the p value, default is 2 for Euclidean distance)
            return np.linalg.norm(embedding1 - embedding2)
        else:
            raise ValueError("Invalid similarity type. Choose from 'cosine', 'mse', 'kl', 'manhattan', 'minkowski'")
    else:
        return 0.0  # or any other value you want to use for NaN or missing values

def process_subfolder(subfolder_path, output_folder):
    input_folder = subfolder_path
    target_size = get_mean_size(input_folder)
    if target_size is not None:
        for gen_image_num in range(0, 5):  # Assuming you have gen_image1.png, gen_image2.png, ..., gen_image5.png
            reference_image_path = os.path.join(input_folder, f'gen_image{gen_image_num}.png')
            reference_embedding = preprocess_and_extract_embedding(reference_image_path, model)

            similarities = []

            for filename in os.listdir(input_folder):
                if is_image_file(filename) and filename != f'gen_image{gen_image_num}.png':
                    image_path = os.path.join(input_folder, filename)
                    other_embedding = preprocess_and_extract_embedding(image_path, model)
                    cosine_similarity = calculate_similarity(reference_embedding, other_embedding, similarity_type='cosine')
                    mse_similarity = calculate_similarity(reference_embedding, other_embedding, similarity_type='mse')
                    kl_similarity = calculate_similarity(reference_embedding, other_embedding, similarity_type='kl')
                    manhattan_distance = calculate_similarity(reference_embedding, other_embedding, similarity_type='manhattan')
                    minkowski_distance = calculate_similarity(reference_embedding, other_embedding, similarity_type='minkowski')
                    similarities.append((filename, cosine_similarity, mse_similarity, kl_similarity, manhattan_distance, minkowski_distance))

            sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

            csv_file_path = os.path.join(output_folder, f'similarities_gen_image{gen_image_num}.csv')

            with open(csv_file_path, mode='w', newline='') as csv_file:
                fieldnames = ['Image', 'Cosine Similarity', 'MSE', 'KL-Div', 'Manhattan Distance', 'Minkowski Distance']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                writer.writeheader()
                for file, cosine_sim, mse_sim, kl_sim, manhattan_dist, minkowski_dist in sorted_similarities:
                    writer.writerow({'Image': file, 'Cosine Similarity': cosine_sim, 'MSE': mse_sim, 'KL-Div': kl_sim, 'Manhattan Distance': manhattan_dist, 'Minkowski Distance': minkowski_dist})

            print(f"Results for gen_image{gen_image_num} saved to {csv_file_path}")

# Update the paths
main_folder = './merged_images'
output_folder_main = './merged_images'

for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    output_folder_sub = os.path.join(output_folder_main, subfolder)

    if os.path.isdir(subfolder_path):
        os.makedirs(output_folder_sub, exist_ok=True)
        process_subfolder(subfolder_path, output_folder_sub)

