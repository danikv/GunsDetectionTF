import os
import shutil
import uuid

mapping_file_name = 'final.dat'

def copy_image(path_to_image, image_class, image_name, mode):
    if not os.path.isdir(os.path.join('data' , mode , image_class)):
        os.mkdir(os.path.join('data', mode , image_class))
    shutil.copyfile(path_to_image, os.path.join('data', mode , image_class, str(uuid.uuid4()) + '.jpg'))

with open(mapping_file_name) as f:
    for line in f:
        splitted_line = line.split(',')
        image_path = f.readline().split(',')[1].strip()
        if image_path == '-':
            continue
        splitted_line.append(image_path)
        if int(splitted_line[1]) == 1:
            copy_image(splitted_line[4], ''.join(splitted_line[3].split()), splitted_line[0], "train")
        else:
            copy_image(splitted_line[4], ''.join(splitted_line[3].split()), splitted_line[0], "test")
