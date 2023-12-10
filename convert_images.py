import os
import random
import SimpleITK as sitk
from PIL import Image
import numpy as np

def mhd_raw_to_jpeg(mhd_file_path, jpeg_file_path, size=(112, 112)):
    print(mhd_file_path)
    # Check if corresponding .raw file exists
    raw_file_path = os.path.splitext(mhd_file_path)[0] + '.raw'
    if not os.path.exists(raw_file_path):
        print(f"Skipping {mhd_file_path} as corresponding .raw file does not exist.")
        return

    itkimage = sitk.ReadImage(mhd_file_path)
    image_array = sitk.GetArrayFromImage(itkimage)
    normalized_image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    image_8bit = (normalized_image * 255).astype(np.uint8)
    image = Image.fromarray(image_8bit)
    image_resized = image.resize(size)
    image_resized.save(jpeg_file_path, 'JPEG')

def process_directory(source_dir, dest_dir, split_ratio=0.1):
    for root, dirs, files in os.walk(source_dir):
        for name in files:
            if name.endswith('.mhd'):
                source_file = os.path.join(root, name)
                relative_path = os.path.relpath(root, source_dir)
                target_subdir = os.path.join(dest_dir, relative_path)

                if not os.path.exists(target_subdir):
                    os.makedirs(target_subdir)

                base_name = os.path.splitext(name)[0]
                target_file = os.path.join(target_subdir, base_name + '.jpg')
                mhd_raw_to_jpeg(source_file, target_file)

def split_data(source_dir, dest_dir, validation_dir, split_ratio):
    for category in ['Diseased', 'Healthy']:
        training_path = os.path.join(source_dir, 'Training', category)
        validation_path = os.path.join(validation_dir, category)
        if not os.path.exists(validation_path):
            os.makedirs(validation_path)

        images = os.listdir(training_path)
        random.shuffle(images)
        validation_count = int(len(images) * split_ratio)

        for i in range(validation_count):
            src = os.path.join(training_path, images[i])
            dest = os.path.join(validation_path, images[i])
            os.rename(src, dest)

# Paths
source_base = 'C:/Users/sidth/Downloads/AD_Classification_data/Classification_data'
dest_base = 'C:/Users/sidth/Downloads/AD_Classification_data/Classification_data_formatted'
validation_base = 'C:/Users/sidth/Downloads/AD_Classification_data/Classification_data_formatted/Validation'

# Process each .mhd file to .jpg
process_directory(source_base, dest_base)

# Split training data into training and validation sets
split_data(dest_base, source_base, validation_base, split_ratio=0.1)
