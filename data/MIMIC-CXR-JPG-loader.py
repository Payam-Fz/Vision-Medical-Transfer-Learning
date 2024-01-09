import tensorflow as tf
import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
from PIL import Image

metadata_csv_file = 'mimic-cxr-2.0.0-metadata.csv.gz'
split_csv_file = 'mimic-cxr-2.0.0-split.csv.gz'
chexpert_csv_file = 'mimic-cxr-2.0.0-chexpert.csv.gz'


class MIMIC_CXR_JPG_Loader:
    def __init__(self, data_folder, csv_folder, split_csv, label_csv):
        self.data_folder = data_folder
        self.metadata_csv = pd.read_csv(os.path.join(csv_folder, metadata_csv_file))
        self.split_csv = pd.read_csv(os.path.join(csv_folder, split_csv_file))
        self.label_csv = pd.read_csv(os.path.join(csv_folder, chexpert_csv_file))
        
    def _load_image(self, subject_id, study_id, dicom_id):  # TODO: usage
        # Implement logic to load images from the provided data_folder
        # You can use any image loading library, e.g., PIL or OpenCV

        image_path = os.path.join(self.data_folder, dicom_id + ".jpg")
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        return image
    
    def _preprocess_labels(self, subject_id, study_id):
        labels = self.label_csv[(self.label_csv['subject_id'] == subject_id) & (self.label_csv['study_id'] == study_id)].iloc[:, 2:]
        return labels.values.flatten()

    def _preprocess_image_label(self, row):
        image = self._load_image(row['subject_id'], row['study_id'], row['dicom_id'])
        labels = self._preprocess_labels(row['subject_id'], row['study_id'])
        return image, labels
    
    def load(self, batch_size=32):
        # Merge metadata with split information
        merged_data = pd.merge(self.metadata_csv, self.split_csv, on=['dicom_id', 'study_id', 'subject_id'])
        
        # Split data into train, validation, and test sets
        train_data, test_data = train_test_split(merged_data, test_size=0.1, stratify=merged_data['split'])
        train_data, val_data = train_test_split(train_data, test_size=0.1, stratify=train_data['split'])
        
        # Define functions for loading and preprocessing data
        def load_and_preprocess_train(row):
            return self._preprocess_image_label(row)
        
        def load_and_preprocess_test(row):
            return self._preprocess_image_label(row)
        
        def load_and_preprocess_val(row):
            return self._preprocess_image_label(row)
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data.to_dict(orient='records'))
        val_dataset = tf.data.Dataset.from_tensor_slices(val_data.to_dict(orient='records'))
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data.to_dict(orient='records'))
        
        # Apply data preprocessing functions
        train_dataset = train_dataset.map(load_and_preprocess_train)
        val_dataset = val_dataset.map(load_and_preprocess_val)
        test_dataset = test_dataset.map(load_and_preprocess_test)
        
        # Define dataset information
        dataset_info = tf.data.Dataset.from_tensor_slices(merged_data.to_dict(orient='records'))
        
        # Return datasets and information
        return train_dataset.batch(batch_size), val_dataset.batch(batch_size), test_dataset.batch(batch_size), dataset_info

# Usage example
data_folder = '/path/to/your/data/folder'
metadata_csv = '/path/to/your/mimic-cxr-2.0.0-metadata.csv.gz'
split_csv = '/path/to/your/mimic-cxr-2.0.0-split.csv.gz'
label_csv = '/path/to/your/mimic-cxr-2.0.0-chexpert.csv.gz'

myCustomDataLoader = MIMIC_CXR_JPG_Loader(data_folder, metadata_csv, split_csv, label_csv)
train_dataset, val_dataset, test_dataset, dataset_info = myCustomDataLoader.load(batch_size=32)

# Accessing information
num_train_images = dataset_info.filter(lambda x: x['split'] == 'train').cardinality().numpy()
num_classes = myCustomDataLoader.label_csv.shape[1] - 2  # excluding subject_id and study_id

# Accessing images and labels
for image, labels in train_dataset.take(1):
    print("Image Shape:", image.shape)
    print("Labels:", labels)
