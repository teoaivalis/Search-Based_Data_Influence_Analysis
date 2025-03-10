import os
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq

def find_top_n_closest_matches(target_text, descriptions, filenames, n=100):
    # Vectorize the descriptions and the target text
    vectorizer = TfidfVectorizer()
    descriptions_vectorized = vectorizer.fit_transform(descriptions)
    target_vectorized = vectorizer.transform([target_text])

    # Calculate cosine similarity between the target text and descriptions
    similarity_scores = cosine_similarity(target_vectorized, descriptions_vectorized)

    # Find the top N closest matches
    closest_matches_indices = heapq.nlargest(n, range(len(similarity_scores[0])), key=similarity_scores[0].__getitem__)

    # Return the top N closest matches and their corresponding filenames
    closest_matches = [(descriptions[i], filenames[i]) for i in closest_matches_indices]

    return closest_matches

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
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Copy each image from the list to the destination folder
    for image in image_list:
        source_path = os.path.join(source_folder, image)
        if os.path.exists(source_path):
            destination_path = os.path.join(destination_folder, image)
            shutil.copyfile(source_path, destination_path)
            print(f"Copied {image} to {destination_folder}")
        else:
            print(f"File {image} not found in {source_folder}")

# Update the paths
folder_path = './Local_Dalle/images/fashion_items'

# Read all text files from the source folder
descriptions, filenames = read_text_files_from_folder(folder_path)

# Generate output.txt file
with open('output.txt', 'w') as output_file:
    for description, filename in zip(descriptions, filenames):
        output_file.write(f"{filename} {description.strip()} {filename[:-4]}.png\n")

# Read contents from output.txt for comparison
with open('output.txt', 'r') as output_file:
    output_contents = output_file.readlines()

# Provide input
target_text = input("Enter target text: ")

# Find the top 5 closest matches
top_n_closest_matches = find_top_n_closest_matches(target_text, descriptions, filenames, n=100)

# Extract the filenames and descriptions from the matches
closest_matches_filenames = [filename for _, filename in top_n_closest_matches]
closest_matches_descriptions = [description for description, _ in top_n_closest_matches]

# Convert the text filenames to PNG filenames
closest_matches_png_filenames = convert_to_png_filenames(closest_matches_filenames)

# Print the list of corresponding PNG filenames along with their descriptions
print("List of corresponding PNG filenames and their descriptions:")
for png_filename, description in zip(closest_matches_png_filenames, closest_matches_descriptions):
    print("PNG Filename:", png_filename)
    print("Description:", description)
    print()  # Add an empty line for better readability

# Print the list of corresponding PNG filenames in the terminal
print("List of corresponding PNG filenames:")
for png_filename in closest_matches_png_filenames:
    print(png_filename)

# Update the paths
destination_folder = f"./{target_text.replace(' ', '_')}"

# Copy the images to the destination folder
copy_images(closest_matches_png_filenames, folder_path, destination_folder)

print(f"Images copied to {destination_folder}")


