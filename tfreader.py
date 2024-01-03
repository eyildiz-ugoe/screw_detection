import tensorflow as tf
import os


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

def save_images(dataset, base_folder, tfrecord_name):
    for i, (image, label) in enumerate(dataset):
        # Create subfolder based on the label
        subfolder = os.path.join(base_folder, f'label_{label.numpy()}')
        os.makedirs(subfolder, exist_ok=True)

        # Save the image to the subfolder with a unique filename
        image_filename = os.path.join(subfolder, f'image_{tfrecord_name}_{i}.jpg')
        try:
            tf.keras.preprocessing.image.save_img(image_filename, image.numpy())
            print(f"Saved image {i} in {subfolder}")
        except Exception as e:
            print(f"Error saving image {i} in {subfolder}: {e}")

def main():
    current_directory = os.getcwd()  # Get the current directory
    base_folder = current_directory  # Use the current directory as the base folder
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
                save_images(dataset, os.path.join(base_folder, folder), os.path.splitext(os.path.basename(tfrecord_file))[0])

if __name__ == "__main__":
    main()
