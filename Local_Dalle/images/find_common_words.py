from collections import Counter 
import os
string = ""
txt_new_list = r"./images/fashion_items/"
for file in os.listdir(txt_new_list):
    old_filepath = os.path.join(txt_new_list, file)
    ext = os.path.splitext(file)[-1].lower()
    #print(old_filepath)
    #print(ext)
    if(ext == ".txt"):
        a_file = open(old_filepath, "r")
        number_of_lines = 1
        for i in range(number_of_lines):
            line = a_file.readline()
            string = string + " " + line
            a_file.close()

print(string)
split_it = string.split() 

# Pass the split_it list to instance of Counter class. 
Counter = Counter(split_it) 
                                                             
# most_common() produces k frequently encountered 
# input values and their respective counts. 
most_occur = Counter.most_common()
new_file = './images/common_words.txt'
with open(new_file, "w") as new_file:
    new_file.write(str(most_occur))
print(most_occur) 
