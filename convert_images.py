import os
import rawpy
import PIL.Image

def convert_raw_to_jpeg_with_metadata(raw_file_path, mhd_file_path, jpeg_file_path):
    # # Read metadata from .mhd file
    # itkimage = sitk.ReadImage(mhd_file_path)
    # # metadata = sitk.GetMetaDataDictionary(itkimage)

    # # Process this metadata as needed...

    # # Read RAW file
    # print(raw_file_path)
    # with rawpy.imread(raw_file_path) as raw:
    #     # Process RAW file (use metadata if necessary)
    #     rgb = raw.postprocess()

    # # Convert to PIL Image and save as JPEG
    # image = PIL.Image.fromarray(rgb)
    # image.save(jpeg_file_path, 'JPEG')

    
    im = PIL.Image.open(raw_file_path)
    rgb_im = im.convert('RGB')
    rgb_im.save(jpeg_file_path)

# Directory containing your .raw and .mhd files
directory = '/scratch/ssb3vk/MLIA/Classification_data/Training/Diseased'

# Iterate over files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.raw'):
        base_name = os.path.splitext(filename)[0]
        raw_file_path = os.path.join(directory, filename)
        mhd_file_path = os.path.join(directory, base_name + '.mhd')
        jpeg_file_path = os.path.join(directory, base_name + '.jpg')

        if os.path.exists(mhd_file_path):
            convert_raw_to_jpeg_with_metadata(raw_file_path, mhd_file_path, jpeg_file_path)