import os

def delete_images_and_descriptions(folder_path, image_list_file):
    # Read the list of selected images from the text file
    with open(image_list_file, 'r') as f:
        selected_images = f.read().splitlines()

    # Delete the selected images and their corresponding description files
    for image in selected_images:
        image_file = os.path.join(folder_path, image)
        description_file = os.path.join(folder_path, image.replace('.jpg', '.txt'))

        # Check if image and description files exist before deleting
        if os.path.exists(image_file):
            os.remove(image_file)
            print(f"Deleted image: {image}")
        else:
            print(f"Image not found: {image}")

        if os.path.exists(description_file):
            os.remove(description_file)
            print(f"Deleted description: {description_file}")
        else:
            print(f"Description not found: {description_file}")

if __name__ == "__main__":
    folder_path = "./Local_Dalle/images/fashion_images"  # Update the path to your dataset folder
    image_list_file = os.path.join(folder_path, "./merged_images/all_selected_images.txt")

    delete_images_and_descriptions(folder_path, image_list_file)

