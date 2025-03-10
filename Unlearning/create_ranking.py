import os
import pandas as pd
import numpy as np

from scipy.optimize import minimize

# Define a function to compute the ranking for a generated image
def compute_ranking(gen_image, folder_path):
    # Construct relative file paths
    raw_comparison_file = os.path.join(folder_path, f"raw_comparison{gen_image}.csv")
    embedding_metrics_file = os.path.join(folder_path, f"similarities_gen_image{gen_image}.csv")
    
    # Check if files exist
    if not os.path.exists(raw_comparison_file) or not os.path.exists(embedding_metrics_file):
        print(f"Error: One or more files not found for {gen_image}. Skipping...")
        return []

    # Read the CSV files
    raw_metrics_df = pd.read_csv(raw_comparison_file)
    embedding_metrics_df = pd.read_csv(embedding_metrics_file)

    # Merge the dataframes
    merged_df = pd.merge(raw_metrics_df, embedding_metrics_df, left_on="Image2", right_on="Image")
    merged_df.to_csv(os.path.join(folder_path, f"{gen_image}_merged_data.csv"), index=False)

    # Read the combined metrics file
    combined_metrics_df = pd.read_csv(os.path.join(folder_path, f"{gen_image}_merged_data.csv"))

    # Define the objective function to be minimized
    def objective_function(weights, similarity_scores):
        combined_scores = combine_similarity_scores(similarity_scores, weights)
        score = sum(combined_scores.values())
        return -score  # Minimize negative of score

    # Combine raw and embedding tuples into similarity_scores dictionary
    similarity_scores = {}
    for _, row in combined_metrics_df.iterrows():
        image_name = row['Image2']
        raw_scores = row[['MSE_x', 'MAE', 'Cosine_Similarity', 'SSIM']].tolist()
        embedding_scores = row[['Cosine Similarity', 'MSE_y', 'KL-Div', 'Manhattan Distance', 'Minkowski Distance']].tolist()

        # Check for NaN values in the similarity scores
        if np.isnan(raw_scores).any() or np.isnan(embedding_scores).any():
            print(f"NaN values found for image {image_name}. Skipping...")
            continue

        similarity_scores[image_name] = raw_scores + embedding_scores

    # Define initial guess for weights
    num_metrics = len(raw_scores) + len(embedding_scores)
    initial_weights = [0.0001, 0.01, -1, -1, -1, 1, 0.00001, 0.00001, 0.001]

    # Minimize objective function to find optimal weights
    best_weights = initial_weights

    # Combine similarity scores using the optimal weights
    combined_scores = combine_similarity_scores(similarity_scores, best_weights)

    # Rank images based on combined similarity scores
    ranked_images = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    # Return the ranked images
    return ranked_images

# Define function to combine similarity scores
def combine_similarity_scores(similarity_scores, weights):
    combined_scores = {}
    for image, scores in similarity_scores.items():
        if np.isnan(scores).any():
            combined_scores[image] = np.nan
            continue
        combined_score = sum(score * weight for score, weight in zip(scores, weights))
        combined_scores[image] = combined_score
    return combined_scores

# Function to iterate over subfolders of a given directory
def process_subfolders(main_folder, num_images=10):
    all_selected_images = []  # List to store selected images from all subfolders
    # Iterate over subfolders
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            print(f"Processing subfolder: {subfolder}")
            selected_images = run_comparison_for_subfolder(subfolder_path, num_images)
            all_selected_images.extend(selected_images)
    # Save the list of all selected images to a text file in the main merged folder
    save_selected_images(main_folder, all_selected_images)
    return all_selected_images

# Function to run the comparison for a specific subfolder and select last num_images images
def run_comparison_for_subfolder(subfolder_path, num_images=10):
    gen_images = ["0", "1", "2", "3", "4"]
    all_rankings = [compute_ranking(gen_image, subfolder_path) for gen_image in gen_images]

    # Combine rankings for all images
    combined_rankings = {}
    for ranking in all_rankings:
        for image, score in ranking:
            if image not in combined_rankings:
                combined_rankings[image] = []
            combined_rankings[image].append(score)

    # Compute mean scores for each image
    mean_scores = {image: np.mean(scores) for image, scores in combined_rankings.items()}

    # Rank images based on mean scores
    final_ranked_images = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)

    # Exclude reference images (0.png, 1.png, 2.png, 3.png, 4.png) from the final ranked images
    final_ranked_images = [image_score for image_score in final_ranked_images if not image_score[0].endswith(".png")]

    # Select last num_images images
    selected_images = [image_score[0] for image_score in final_ranked_images[-num_images:]]

    # Return the list of selected images in the subfolder
    return selected_images

# Function to save list of selected images to a text file in the main merged folder
def save_selected_images(main_folder, selected_images):
    with open(os.path.join(main_folder, "all_selected_images.txt"), "w") as file:
        for image in selected_images:
            file.write(image + "\n")

# Main function to start processing
def main():
    # Update with your paths
    main_folder = "./merged_images"
    num_images_per_subfolder = 10
    all_selected_images = process_subfolders(main_folder, num_images_per_subfolder)
    print("Total selected images:", len(all_selected_images))

if __name__ == "__main__":
    main()

