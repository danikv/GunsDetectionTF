import os
import shutil

model_file_name = 'data/images_family_val.txt'

models = {}
with open(model_file_name) as f:
    for line in f:
       splitted_line = line.split()
       models[splitted_line[0]] = ''.join(splitted_line[1:])

for key, value in models.items():
    if not os.path.isdir(os.path.join('data/val/', value)):
        os.mkdir(os.path.join('data/val/', value))
    image_name = key + '.jpg'
    shutil.copyfile(os.path.join('data/images/', image_name), os.path.join('data/val/', value, image_name))
