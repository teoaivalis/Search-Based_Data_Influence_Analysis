import os
import sys
import shutil
import subprocess
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to generate images
def generate_images(text_input, num_images=5):
    command = [
        'python3',
        './Local_Dalle/generate.py',
        '--dalle_path', './Local_Dalle/fashion_16_12_30ep.pt',
        '--text', text_input,
        '--num_images', str(num_images),
        '--taming'
    ]
    
    try:
        subprocess.run(command, check=True)
        print("Image generation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

# Function to read all text files from a folder
def read_text_files_from_folder(folder_path):
    descriptions = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                description = file.read()
                descriptions.append(description)
                filenames.append(filename)
    return descriptions, filenames

# Function to convert text filenames to PNG filenames
def convert_to_png_filenames(txt_filenames):
    png_filenames = [filename[:-4] + ".jpg" for filename in txt_filenames]
    return png_filenames

# Function to copy images to a destination folder
def copy_images(image_list, source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for image in image_list:
        source_path = os.path.join(source_folder, image)
        if os.path.exists(source_path):
            destination_path = os.path.join(destination_folder, image)
            shutil.copyfile(source_path, destination_path)
            print(f"Copied {image} to {destination_folder}")
        else:
            print(f"File {image} not found in {source_folder}")

# Function to find top N closest matches
def find_top_n_closest_matches(target_text, descriptions, filenames, n=100):
    vectorizer = TfidfVectorizer()
    descriptions_vectorized = vectorizer.fit_transform(descriptions)
    target_vectorized = vectorizer.transform([target_text])

    similarity_scores = cosine_similarity(target_vectorized, descriptions_vectorized)

    closest_matches_indices = heapq.nlargest(n, range(len(similarity_scores[0])), key=similarity_scores[0].__getitem__)

    closest_matches = [(descriptions[i], filenames[i]) for i in closest_matches_indices]

    return closest_matches

# Function to merge folders
def merge_folders(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        if os.path.isdir(source_path):
            # Recursively merge subfolders
            merge_folders(source_path, destination_path)
        else:
            # Check if source and destination are different
            if source_path != destination_path:
                # Copy files
                shutil.copy2(source_path, destination_path)
                print(f"Copied {item} to {destination_folder}")

    print("Images merged successfully.")

# Function to pair folders based on common prefix
def pair_folders(merged_folder, num_common_words=5):
    folder_pairs = {}
    for folder in os.listdir(merged_folder):
        name_parts = folder.split('_')
        prefix = '_'.join(name_parts[:num_common_words])
        if prefix not in folder_pairs:
            folder_pairs[prefix] = []
        folder_pairs[prefix].append(os.path.join(merged_folder, folder))
    return folder_pairs

# Main function
def main(argv):
    if len(argv) != 2:
        print("Usage: python3 full_experiment.py <descriptions_file>")
        return
    # Updatre with the correct paths
    prompts_file_path = argv[1]
    num_images_to_generate = 5 
    folder_path = '/Local_Dalle/images/fashion_items'
    gen_images_folder = './Local_Dalle/outputs'
    retrieved_images_folder = './Local_Dalle/retrieved_images'
    merged_folder = './Local_Dalle/merged_images'

    # Read prompts from file
    with open(prompts_file_path, 'r') as file:
        prompts = file.readlines()
    prompts = [prompt.strip() for prompt in prompts]

    for idx, prompt in enumerate(prompts, start=1):
        print(f"Generating images for prompt {idx}...")
        generate_images(prompt, num_images_to_generate)

        # Read all text files from the source folder
        descriptions, filenames = read_text_files_from_folder(folder_path)

        # Find the top 5 closest matches
        top_n_closest_matches = find_top_n_closest_matches(prompt, descriptions, filenames, n=100)

        # Extract the filenames from the matches
        closest_matches_filenames = [filename for _, filename in top_n_closest_matches]

        # Convert the text filenames to PNG filenames
        closest_matches_png_filenames = convert_to_png_filenames(closest_matches_filenames)

        # Create the destination folder based on the prompt
        destination_folder = os.path.join(retrieved_images_folder, prompt.replace(' ', '_'))

        # Copy the images to the destination folder
        copy_images(closest_matches_png_filenames, folder_path, destination_folder)

        print(f"Images copied to {destination_folder}")

    # Merge the generated images folder and the retrieved images folder
    merge_folders(gen_images_folder, merged_folder)
    merge_folders(retrieved_images_folder, merged_folder)

    print("Images merged successfully.")

    # Pair folders within merged_folder
    pairs = pair_folders(merged_folder)
    for prefix, folders in pairs.items():
        if len(folders) >= 2:  # Check if there are at least two folders for merging
            merged_images_folder = os.path.join(merged_folder, f"{prefix}_merged")
            for folder in folders:
                merge_folders(folder, merged_images_folder)
                print(f"Images of paired folders merged successfully in {merged_images_folder}")

if __name__ == "__main__":
    main(sys.argv)

