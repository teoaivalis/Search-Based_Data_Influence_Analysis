import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import csv
import imghdr
import shutil
from sklearn.metrics.pairwise import cosine_similarity

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def resize_images(input_folder, output_folder, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        
        # Skip non-file entities
        if not os.path.isfile(img_path):
            print(f"Skipping non-file entity: {filename}")
            continue
        
        # Skip non-image files
        if not imghdr.what(img_path):
            print(f"Skipping non-image file: {filename}")
            continue
        
        img = cv2.imread(img_path)

        if img is not None:
            resized_img = cv2.resize(img, (target_size[0], target_size[1]))
            output_path = os.path.join(output_folder, filename)
            print("Output Path:", output_path)
            try:
                cv2.imwrite(output_path, resized_img)
            except cv2.error as e:
                print(f"OpenCV error: {e}")
                continue


def get_mean_size(input_folder):
    sizes = []

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            sizes.append((img.shape[1], img.shape[0]))

    if sizes:
        mean_size = np.mean(sizes, axis=0)
        return mean_size.astype(int)
    else:
        return None


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err


def calculate_ssim(imageA, imageB):
    subtracted = cv2.subtract(imageA, imageB)
    subtracted_gray = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(subtracted_gray, np.zeros_like(subtracted_gray))
    return ssim_value


def compare_images(imageA, imageB, title):
    if imageA is not None and imageB is not None:
        error = mse(imageA, imageB)
        mae_value = np.mean(np.abs(imageA.astype("float") - imageB.astype("float")))
        cross_correlation = np.sum((imageA.astype("float") - np.mean(imageA)) * (
                    imageB.astype("float") - np.mean(imageB))) / (
                                           imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
        hist_comparison = cv2.compareHist(cv2.calcHist([imageA], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]),
                                           cv2.calcHist([imageB], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]),
                                           cv2.HISTCMP_CORREL)
        cosine_sim = cosine_similarity(imageA.flatten().reshape(1, -1), imageB.flatten().reshape(1, -1))[0][0]
        s = calculate_ssim(imageA, imageB)

        return error, mae_value, cross_correlation, hist_comparison, cosine_sim, s
    else:
        return None, None, None, None, None, None


def write_results_to_csv(results, csv_filename):
    with open(csv_filename, mode='w', newline='') as csvfile:
        fieldnames = ['Image1', 'Image2', 'MSE', 'MAE', 'Cross_Correlation', 'Histogram_Comparison',
                      'Cosine_Similarity', 'SSIM']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)


def process_subfolders(main_folder, destination_folder, error_folder):
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        print(subfolder_path)
        if os.path.isdir(subfolder_path) and not subfolder.startswith('.'):
            input_folder = subfolder_path
            output_folder = os.path.join(subfolder_path, "output")
            csv_filename = os.path.join(subfolder_path, "raw_comparison5.csv")

            mean_size = get_mean_size(input_folder)

            if mean_size is not None:
                resize_images(input_folder, output_folder, mean_size)

                reference_filename = "05.png"
                reference_path = os.path.join(output_folder, reference_filename)
                reference_image = cv2.imread(reference_path)

                # Check if raw_comparison.csv already exists
                if os.path.exists(csv_filename):
                    print(f"Skipping comparison for {subfolder} as raw_comparison.csv already exists")

                    # Remove the output folder
                    shutil.rmtree(output_folder)

                    # Move the processed subfolder to destination_folder
                    shutil.move(subfolder_path, os.path.join(destination_folder, subfolder))
                    print(f"Moved {subfolder} to {destination_folder}")

                else:
                    print(f"Creating {csv_filename}")

                    results = []

                    for filename in os.listdir(output_folder):
                        img_path = os.path.join(output_folder, filename)

                        if os.path.isfile(img_path) and imghdr.what(img_path) is None:
                            print(f"Skipping non-image file: {filename}")
                            continue

                        img = cv2.imread(img_path)
                        if img is not None and filename != reference_filename:
                            error, mae_value, cross_correlation, hist_comparison, cosine_sim, ssim_value = \
                                compare_images(reference_image, img, f"{reference_filename} vs. {filename}")

                            if error is None:
                                print(f"Error processing {filename}. Moving folder to error folder.")
                                shutil.move(subfolder_path, os.path.join(error_folder, subfolder))
                                break

                            results.append({
                                'Image1': reference_filename,
                                'Image2': filename,
                                'MSE': error,
                                'MAE': mae_value,
                                'Cross_Correlation': cross_correlation,
                                'Histogram_Comparison': hist_comparison,
                                'Cosine_Similarity': cosine_sim,
                                'SSIM': ssim_value
                            })

                    if len(results) > 0:
                        results = sorted(results, key=lambda x: x['SSIM'], reverse=True)

                        # Write results to raw_comparison.csv
                        write_results_to_csv(results, csv_filename)
                        print(f"Comparison results written to {csv_filename}")

                        # Remove the output folder
                        shutil.rmtree(output_folder)

                        # Move the processed subfolder to destination_folder
                        shutil.move(subfolder_path, os.path.join(destination_folder, subfolder))
                        print(f"Moved {subfolder} to {destination_folder}")

            else:
                print(f"No images found in {subfolder}. Moving folder to error folder.")
                shutil.move(subfolder_path, os.path.join(error_folder, subfolder))
    
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        print(subfolder_path)
        if os.path.isdir(subfolder_path) and not subfolder.startswith('.'):
            input_folder = subfolder_path
            output_folder = os.path.join(subfolder_path, "output")
            csv_filename = os.path.join(subfolder_path, "raw_comparison4.csv")

            mean_size = get_mean_size(input_folder)

            if mean_size is not None:
                resize_images(input_folder, output_folder, mean_size)

                reference_filename = "04.png"
                reference_path = os.path.join(output_folder, reference_filename)
                reference_image = cv2.imread(reference_path)

                # Check if raw_comparison.csv already exists
                if os.path.exists(csv_filename):
                    print(f"Skipping comparison for {subfolder} as raw_comparison.csv already exists")

                    # Remove the output folder
                    shutil.rmtree(output_folder)

                    # Move the processed subfolder to destination_folder
                    shutil.move(subfolder_path, os.path.join(destination_folder, subfolder))
                    print(f"Moved {subfolder} to {destination_folder}")

                else:
                    print(f"Creating {csv_filename}")

                    results = []

                    for filename in os.listdir(output_folder):
                        img_path = os.path.join(output_folder, filename)

                        if os.path.isfile(img_path) and imghdr.what(img_path) is None:
                            print(f"Skipping non-image file: {filename}")
                            continue

                        img = cv2.imread(img_path)
                        if img is not None and filename != reference_filename:
                            error, mae_value, cross_correlation, hist_comparison, cosine_sim, ssim_value = \
                                compare_images(reference_image, img, f"{reference_filename} vs. {filename}")

                            if error is None:
                                print(f"Error processing {filename}. Moving folder to error folder.")
                                shutil.move(subfolder_path, os.path.join(error_folder, subfolder))
                                break

                            results.append({
                                'Image1': reference_filename,
                                'Image2': filename,
                                'MSE': error,
                                'MAE': mae_value,
                                'Cross_Correlation': cross_correlation,
                                'Histogram_Comparison': hist_comparison,
                                'Cosine_Similarity': cosine_sim,
                                'SSIM': ssim_value
                            })

                    if len(results) > 0:
                        results = sorted(results, key=lambda x: x['SSIM'], reverse=True)

                        # Write results to raw_comparison.csv
                        write_results_to_csv(results, csv_filename)
                        print(f"Comparison results written to {csv_filename}")

                        # Remove the output folder
                        shutil.rmtree(output_folder)

                        # Move the processed subfolder to destination_folder
                        shutil.move(subfolder_path, os.path.join(destination_folder, subfolder))
                        print(f"Moved {subfolder} to {destination_folder}")

            else:
                print(f"No images found in {subfolder}. Moving folder to error folder.")
                shutil.move(subfolder_path, os.path.join(error_folder, subfolder))

    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        print(subfolder_path)
        if os.path.isdir(subfolder_path) and not subfolder.startswith('.'):
            input_folder = subfolder_path
            output_folder = os.path.join(subfolder_path, "output")
            csv_filename = os.path.join(subfolder_path, "raw_comparison3.csv")

            mean_size = get_mean_size(input_folder)

            if mean_size is not None:
                resize_images(input_folder, output_folder, mean_size)

                reference_filename = "03.png"
                reference_path = os.path.join(output_folder, reference_filename)
                reference_image = cv2.imread(reference_path)

                # Check if raw_comparison.csv already exists
                if os.path.exists(csv_filename):
                    print(f"Skipping comparison for {subfolder} as raw_comparison.csv already exists")

                    # Remove the output folder
                    shutil.rmtree(output_folder)

                    # Move the processed subfolder to destination_folder
                    shutil.move(subfolder_path, os.path.join(destination_folder, subfolder))
                    print(f"Moved {subfolder} to {destination_folder}")

                else:
                    print(f"Creating {csv_filename}")

                    results = []

                    for filename in os.listdir(output_folder):
                        img_path = os.path.join(output_folder, filename)

                        if os.path.isfile(img_path) and imghdr.what(img_path) is None:
                            print(f"Skipping non-image file: {filename}")
                            continue

                        img = cv2.imread(img_path)
                        if img is not None and filename != reference_filename:
                            error, mae_value, cross_correlation, hist_comparison, cosine_sim, ssim_value = \
                                compare_images(reference_image, img, f"{reference_filename} vs. {filename}")

                            if error is None:
                                print(f"Error processing {filename}. Moving folder to error folder.")
                                shutil.move(subfolder_path, os.path.join(error_folder, subfolder))
                                break

                            results.append({
                                'Image1': reference_filename,
                                'Image2': filename,
                                'MSE': error,
                                'MAE': mae_value,
                                'Cross_Correlation': cross_correlation,
                                'Histogram_Comparison': hist_comparison,
                                'Cosine_Similarity': cosine_sim,
                                'SSIM': ssim_value
                            })

                    if len(results) > 0:
                        results = sorted(results, key=lambda x: x['SSIM'], reverse=True)

                        # Write results to raw_comparison.csv
                        write_results_to_csv(results, csv_filename)
                        print(f"Comparison results written to {csv_filename}")

                        # Remove the output folder
                        shutil.rmtree(output_folder)

                        # Move the processed subfolder to destination_folder
                        shutil.move(subfolder_path, os.path.join(destination_folder, subfolder))
                        print(f"Moved {subfolder} to {destination_folder}")

            else:
                print(f"No images found in {subfolder}. Moving folder to error folder.")
                shutil.move(subfolder_path, os.path.join(error_folder, subfolder))

    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        print(subfolder_path)
        if os.path.isdir(subfolder_path) and not subfolder.startswith('.'):
            input_folder = subfolder_path
            output_folder = os.path.join(subfolder_path, "output")
            csv_filename = os.path.join(subfolder_path, "raw_comparison2.csv")

            mean_size = get_mean_size(input_folder)

            if mean_size is not None:
                resize_images(input_folder, output_folder, mean_size)

                reference_filename = "02.png"
                reference_path = os.path.join(output_folder, reference_filename)
                reference_image = cv2.imread(reference_path)

                # Check if raw_comparison.csv already exists
                if os.path.exists(csv_filename):
                    print(f"Skipping comparison for {subfolder} as raw_comparison.csv already exists")

                    # Remove the output folder
                    shutil.rmtree(output_folder)

                    # Move the processed subfolder to destination_folder
                    shutil.move(subfolder_path, os.path.join(destination_folder, subfolder))
                    print(f"Moved {subfolder} to {destination_folder}")

                else:
                    print(f"Creating {csv_filename}")

                    results = []

                    for filename in os.listdir(output_folder):
                        img_path = os.path.join(output_folder, filename)

                        if os.path.isfile(img_path) and imghdr.what(img_path) is None:
                            print(f"Skipping non-image file: {filename}")
                            continue

                        img = cv2.imread(img_path)
                        if img is not None and filename != reference_filename:
                            error, mae_value, cross_correlation, hist_comparison, cosine_sim, ssim_value = \
                                compare_images(reference_image, img, f"{reference_filename} vs. {filename}")

                            if error is None:
                                print(f"Error processing {filename}. Moving folder to error folder.")
                                shutil.move(subfolder_path, os.path.join(error_folder, subfolder))
                                break

                            results.append({
                                'Image1': reference_filename,
                                'Image2': filename,
                                'MSE': error,
                                'MAE': mae_value,
                                'Cross_Correlation': cross_correlation,
                                'Histogram_Comparison': hist_comparison,
                                'Cosine_Similarity': cosine_sim,
                                'SSIM': ssim_value
                            })

                    if len(results) > 0:
                        results = sorted(results, key=lambda x: x['SSIM'], reverse=True)

                        # Write results to raw_comparison.csv
                        write_results_to_csv(results, csv_filename)
                        print(f"Comparison results written to {csv_filename}")

                        # Remove the output folder
                        shutil.rmtree(output_folder)

                        # Move the processed subfolder to destination_folder
                        shutil.move(subfolder_path, os.path.join(destination_folder, subfolder))
                        print(f"Moved {subfolder} to {destination_folder}")

            else:
                print(f"No images found in {subfolder}. Moving folder to error folder.")
                shutil.move(subfolder_path, os.path.join(error_folder, subfolder))


    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        print(subfolder_path)
        if os.path.isdir(subfolder_path) and not subfolder.startswith('.'):
            input_folder = subfolder_path
            output_folder = os.path.join(subfolder_path, "output")
            csv_filename = os.path.join(subfolder_path, "raw_comparison1.csv")

            mean_size = get_mean_size(input_folder)

            if mean_size is not None:
                resize_images(input_folder, output_folder, mean_size)

                reference_filename = "01.png"
                reference_path = os.path.join(output_folder, reference_filename)
                reference_image = cv2.imread(reference_path)

                # Check if raw_comparison.csv already exists
                if os.path.exists(csv_filename):
                    print(f"Skipping comparison for {subfolder} as raw_comparison.csv already exists")

                    # Remove the output folder
                    shutil.rmtree(output_folder)

                    # Move the processed subfolder to destination_folder
                    shutil.move(subfolder_path, os.path.join(destination_folder, subfolder))
                    print(f"Moved {subfolder} to {destination_folder}")

                else:
                    print(f"Creating {csv_filename}")

                    results = []

                    for filename in os.listdir(output_folder):
                        img_path = os.path.join(output_folder, filename)

                        if os.path.isfile(img_path) and imghdr.what(img_path) is None:
                            print(f"Skipping non-image file: {filename}")
                            continue

                        img = cv2.imread(img_path)
                        if img is not None and filename != reference_filename:
                            error, mae_value, cross_correlation, hist_comparison, cosine_sim, ssim_value = \
                                compare_images(reference_image, img, f"{reference_filename} vs. {filename}")

                            if error is None:
                                print(f"Error processing {filename}. Moving folder to error folder.")
                                shutil.move(subfolder_path, os.path.join(error_folder, subfolder))
                                break

                            results.append({
                                'Image1': reference_filename,
                                'Image2': filename,
                                'MSE': error,
                                'MAE': mae_value,
                                'Cross_Correlation': cross_correlation,
                                'Histogram_Comparison': hist_comparison,
                                'Cosine_Similarity': cosine_sim,
                                'SSIM': ssim_value
                            })

                    if len(results) > 0:
                        results = sorted(results, key=lambda x: x['SSIM'], reverse=True)

                        # Write results to raw_comparison.csv
                        write_results_to_csv(results, csv_filename)
                        print(f"Comparison results written to {csv_filename}")

                        # Remove the output folder
                        shutil.rmtree(output_folder)

                        # Move the processed subfolder to destination_folder
                        shutil.move(subfolder_path, os.path.join(destination_folder, subfolder))
                        print(f"Moved {subfolder} to {destination_folder}")

            else:
                print(f"No images found in {subfolder}. Moving folder to error folder.")
                shutil.move(subfolder_path, os.path.join(error_folder, subfolder))

if __name__ == "__main__":
    #Update with the correct paths
    main_folder = "./merged_images/"
    destination_folder = "./merged_images"
    error_folder = "./merged_images"
    process_subfolders(main_folder, destination_folder, error_folder)

