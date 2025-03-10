from os.path import basename
import os
from glob import glob
import shutil


txt_list = r"./images/fashion_items"
txt_new_list = r"./images/fashion_items"
max_length = 512
for file in os.listdir(txt_list):
    old_filepath = os.path.join(txt_list, file)
    ext = os.path.splitext(file)[-1].lower()
    print(old_filepath)
    print(ext)
    if(ext == ".txt"):
        current_length = 0
        a_file = open(old_filepath, "r")
        new_name = file
        new_filepath = os.path.join(txt_new_list, new_name)
        number_of_lines = 2
        with open(new_filepath, "w") as new_file:
            for i in range(number_of_lines):
                line = a_file.readline()
                current_length += len(line)
                if(current_length < max_length):
                    new_file.write(line)
                else:
                    current_length -= len(line)
        a_file.close()
        #os.remove(old_filepath)
    else:
        new_file = file
        new_filepath = os.path.join(txt_new_list, new_file)
        shutil.move(old_filepath,new_filepath)

