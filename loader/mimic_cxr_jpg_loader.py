import tensorflow as tf
import pandas as pd
import numpy as np
import os
import gzip
from sklearn.model_selection import train_test_split
from PIL import Image

data_folder = '../data/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
csv_folder = '../data/physionet.org/files/mimic-cxr-jpg/2.0.0'
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
    # split_size is of form: {'train': int, 'validate': int, 'test': int}
    def __init__(self, split_size={}):
        self._in_split_size = split_size
        self.metadata = {}
        
    # def _load_image(self, subject_id, study_id, dicom_id):
    #     image_path = os.path.join(self.data_folder, subject_id[:3], subject_id, study_id, dicom_id + ".jpg")
    #     image = Image.open(image_path)
    #     image = image.convert("RGB")
    #     image = np.array(image) / 255.0  # Normalize to [0, 1]
    #     return image
        
    def _load_image(self, subject_id, study_id, dicom_id):
        print('subject_id', subject_id)
        image_path = os.path.join(data_folder, subject_id[:3], subject_id, study_id, dicom_id + ".jpg")
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=1, dtype=tf.float32)
        return image

    def _load_label(self, subject_id, study_id):
        label_df = self.label_csv[(self.label_csv['subject_id'] == subject_id) & (self.label_csv['study_id'] == study_id)].iloc[:, 2:]
        assert(label_df.shape[0] == 1)
        # Multi-hot encode labels (using explicit column names to prevent errors due to reordered columns)
        multi_hot_labels = [label_mapping[str(label_df.loc[0, col])] for col in label_columns]
        # one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_columns))
        return multi_hot_labels

    def _preprocess_image_label(self, row):
        print('row:', row)
        with tf.compat.v1.Session() as sess:
            _row = sess.run(row)
        
        image = self._load_image(_row['subject_id'], _row['study_id'], _row['dicom_id'])
        label = self._load_label(_row['subject_id'], _row['study_id'])
        info = _row
        return image, label, info
    
    def load(self):
        # Read .csv info files
        with gzip.open(os.path.join(csv_folder, chexpert_csv_file), 'rt') as file:
            self.label_csv = pd.read_csv(file)
        with gzip.open(os.path.join(csv_folder, metadata_csv_file), 'rt') as file:
            metadata_csv = pd.read_csv(file)
        with gzip.open(os.path.join(csv_folder, split_csv_file), 'rt') as file:
            split_csv = pd.read_csv(file)
            
        # Merge the data
        merged_data = pd.merge(metadata_csv, split_csv, on=['dicom_id', 'study_id', 'subject_id'])
        self.metadata['total_size'] = merged_data.shape[0] 

        # Group data by 'split' column and get sizes
        grouped_data = merged_data.groupby('split', group_keys=False)
        csv_split_size = grouped_data.size().to_dict()
        
        # Split data into train, validation, and test sets and apply sampling based on split_size
        train_split = grouped_data.get_group('train')
        if ('train' in self._in_split_size) & (self._in_split_size['train'] < csv_split_size['train']):
            train_split = train_split.sample(n=self._in_split_size['train'], random_state=42)
        val_split = grouped_data.get_group('validate')
        if ('validate' in self._in_split_size) & (self._in_split_size['validate'] < csv_split_size['validate']):
            val_split = val_split.sample(n=self._in_split_size['validate'], random_state=42)
        test_split = grouped_data.get_group('test')
        if ('test' in self._in_split_size) & (self._in_split_size['test'] < csv_split_size['test']):
            test_split = test_split.sample(n=self._in_split_size['test'], random_state=42)

        self.metadata['split_size'] = { 'train': train_split.shape[0], 'validate': val_split.shape[0], 'test': test_split.shape[0] }
        self.metadata['split_size_frac'] = {key: value / self.metadata['total_size'] for key, value in self.metadata['split_size'].items()}
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(train_split.fillna('').to_dict(orient='list'))
        val_dataset = tf.data.Dataset.from_tensor_slices(val_split.fillna('').to_dict(orient='list'))
        test_dataset = tf.data.Dataset.from_tensor_slices(test_split.fillna('').to_dict(orient='list'))
        train_dataset = train_dataset.map(self._preprocess_image_label)
        val_dataset = val_dataset.map(self._preprocess_image_label)
        test_dataset = test_dataset.map(self._preprocess_image_label)
        
        return train_dataset, val_dataset, test_dataset
