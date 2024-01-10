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
    # split_sizes is of form: {'train': int, 'validate': int, 'test': int}
    def __init__(self, data_folder, csv_folder, split_sizes):
        self.data_folder = data_folder
        self.split_sizes = split_sizes
        self.load()
        
    def _load_image(self, subject_id, study_id, dicom_id):
        image_path = os.path.join(self.data_folder, subject_id[:3], subject_id, study_id, dicom_id + ".jpg")
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        return image

    def _unify_labels(row):
        # Map labels to numerical values (1.0, -1.0, 0.0)
        label_mapping = {'1.0': 1.0, '-1.0': -1.0, '0.0': 0.0, 'missing': np.nan}
        
        # Extract label columns from the row
        label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                        'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
                        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
        
        # Convert labels to numerical values
        labels = [label_mapping[str(row[col])] for col in label_columns]
        
        # One-hot encode labels
        one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_columns))
        
        return one_hot_labels
    
    def _preprocess_labels(self, subject_id, study_id):
        labels = self.label_csv[(self.label_csv['subject_id'] == subject_id) & (self.label_csv['study_id'] == study_id)].iloc[:, 2:]
        return labels.values.flatten()

    def _preprocess_image_label(self, row):
        image = self._load_image(row['subject_id'], row['study_id'], row['dicom_id'])
        labels = self._preprocess_labels(row['subject_id'], row['study_id'])
        return image, labels
    
    def load(self, batch_size=32):
        self.label_csv = pd.read_csv(os.path.join(self.csv_folder, chexpert_csv_file))
        metadata_csv = pd.read_csv(os.path.join(self.csv_folder, metadata_csv_file))
        split_csv = pd.read_csv(os.path.join(self.csv_folder, split_csv_file))
        merged_metadata = pd.merge(metadata_csv, split_csv, on=['dicom_id', 'study_id', 'subject_id'])

        # Split data into train, validation, and test sets and apply sampling based on split_size
        grouped_data = merged_metadata.groupby('split', group_keys=False)
        train_data = grouped_data[grouped_data['split'] == 'train'].sample(n=self.split_size['train'], random_state=42)
        val_data = grouped_data[grouped_data['split'] == 'validate'].sample(n=self.split_size['validate'], random_state=42)
        test_data = grouped_data[grouped_data['split'] == 'test'].sample(n=self.split_size['test'], random_state=42)

        
        
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
