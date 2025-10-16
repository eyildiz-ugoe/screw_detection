"""
Convert TFRecord files from ScrewDTF dataset to JPG images.

This script converts the TFRecord format dataset (downloaded from Zenodo)
into organized JPG images in label_0 and label_1 folders.

Usage:
    python scripts/convert_tfrecords.py [path_to_ScrewDTF_folder]

If no path is provided, it will look for ScrewDTF/ in the current directory.
"""
import tensorflow as tf
import os
import sys

def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    label = tf.cast(example['label'], tf.int32)
    return image, label

def load_dataset(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    return parsed_dataset

def save_images(dataset, base_folder):
    for i, (image, label) in enumerate(dataset):
        # Create subfolder based on the label
        subfolder = os.path.join(base_folder, f'label_{label.numpy()}')
        os.makedirs(subfolder, exist_ok=True)

        # Save the image to the subfolder with a unique filename
        image_filename = os.path.join(subfolder, f'image_{i}.jpg')
        try:
            tf.keras.preprocessing.image.save_img(image_filename, image.numpy())
            print(f"Saved image {i} in {subfolder}")
        except Exception as e:
            print(f"Error saving image {i} in {subfolder}: {e}")

def main():
    # Allow user to specify ScrewDTF folder path, or use default
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
    else:
        base_folder = os.path.join(os.getcwd(), 'ScrewDTF')
    
    if not os.path.exists(base_folder):
        print(f"Error: Folder '{base_folder}' not found!")
        print("Usage: python scripts/convert_tfrecords.py [path_to_ScrewDTF_folder]")
        sys.exit(1)
    
    print(f"Converting TFRecords in: {base_folder}")
    folders = ['Test', 'Eval', 'Train']

    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        tfrecord_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.tfrecord')]

        print(f"TFRecord files found in {folder} folder:")
        print(tfrecord_files)  # Print the list of TFRecord file paths

        for tfrecord_file in tfrecord_files:
            print(f"Processing TFRecord file {tfrecord_file}:")
            
            # Load and process the TFRecord file
            dataset = load_dataset(tfrecord_file)

            if dataset is not None:
                # Save the images to subfolders based on labels
                save_images(dataset, os.path.join(base_folder, folder))

if __name__ == "__main__":
    main()
