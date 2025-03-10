#!/bin/bash


source environment/bin/activate

mkdir ./Insights_from_the_dataset/Unlearning/merged_images_1
mkdir ./Insights_from_the_dataset/Unlearning/retrieved_images_1
python3 ./Insights_from_the_dataset/Unlearning/full_experiment.py ./Insights_from_the_dataset/Unlearning/descriptions/descriptions_1.txt

python3 ./Insights_from_the_dataset/Unlearning/compare_embeddings.py
python3 ./Insights_from_the_dataset/Unlearning/raw_image_similarity.py
python3 ./Insights_from_the_dataset/Unlearning/create_ranking.py
unzip ./Insights_from_the_dataset/Local_Dalle/images/tmp.zip -d ./Insights_from_the_dataset/Local_Dalle/images
python3 ./Insights_from_the_dataset/Unlearning/remove_from_dataset.py

python3 ./Insights_from_the_dataset/Local_Model/train_dalle.py
