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
        
# Turn labels into multi-hot-encoding
# 1.0 : The label was positively mentioned in the associated study, and is present in one or more of the corresponding images
# 0.0 : The label was negatively mentioned in the associated study, and therefore should not be present in any of the corresponding images
# -1.0 : The label was either:
#   (1) mentioned with uncertainty in the report, and therefore may or may not be present to some degree in the corresponding image, or
#   (2) mentioned with ambiguous language in the report and it is unclear if the pathology exists or not
# Missing (empty element) : No mention of the label was made in the report
label_mapping = {'1': 1, '-1': 0, '0': 0, '': 0}

# Order matters
label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
                'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']


class MIMIC_CXR_JPG_Loader:
    # split_sizes is of form: {'train': int, 'validate': int, 'test': int}
    def __init__(self, data_folder, csv_folder, split_sizes):
        self.data_folder = data_folder
        self.csv_folder = csv_folder
        self.split_sizes = split_sizes
        self.load()
        
    def _load_image(self, subject_id, study_id, dicom_id):
        image_path = os.path.join(self.data_folder, subject_id[:3], subject_id, study_id, dicom_id + ".jpg")
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        return image

    def _load_label(self, subject_id, study_id):
        label_df = self.label_csv[(self.label_csv['subject_id'] == subject_id) & (self.label_csv['study_id'] == study_id)].iloc[:, 2:]
        assert(label_df.shape[0] == 1)
        
        # One-hot encode labels (using explicit column names to prevent errors due to reordered columns)
        multi_hot_labels = [label_mapping[str(labels_df.loc[0, col])] for col in label_columns]
        # one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_columns))
        return multi_hot_labels

    def _preprocess_image_label(self, row):
        image = self._load_image(row['subject_id'], row['study_id'], row['dicom_id'])
        labels = self._load_label(row['subject_id'], row['study_id'])
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
        
        # Apply data preprocessing functions
        train_dataset = train_dataset.map(self._preprocess_image_label)
        val_dataset = val_dataset.map(self._preprocess_image_label)
        test_dataset = test_dataset.map(self._preprocess_image_label)
        
        # ISSUE: map is loading the imges
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data.to_dict(orient='records'))
        val_dataset = tf.data.Dataset.from_tensor_slices(val_data.to_dict(orient='records'))
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data.to_dict(orient='records'))
        
        # Define dataset information
        dataset_info = tf.data.Dataset.from_tensor_slices(merged_data.to_dict(orient='records'))
        
        # Return datasets and information
        return train_dataset.batch(batch_size), val_dataset.batch(batch_size), test_dataset.batch(batch_size), dataset_info

# Usage example
data_folder = '../data/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
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
