import tensorflow_hub as hub
from utils.augmentation import preprocess_image
from loader.mimic_cxr_jpg_loader import MIMIC_CXR_JPG_Loader
from utils.analysis import *
from IPython.display import Image
from loader.labels import LABELS
import matplotlib.pyplot as plt
import numpy as np
import os

BATCH_SIZE = 10

# VARIABLES
project_folder = os.getcwd()
if project_folder.endswith('/job'):
  project_folder = project_folder[:-4]
BASE_MODEL_PATH = './base-models/remedis/cxr-152x2-remedis-m/'
hub_path = os.path.join(project_folder, BASE_MODEL_PATH)
IMAGE_SIZE = (448, 448)
START_TIME = get_curr_datetime()

# Load hub
module = hub.load(hub_path)

# Pathology: The image is of shape (<BATCH_SIZE>, 224, 224, 3)
# Chest X-Ray: The image is of shape (<BATCH_SIZE>, 448, 448, 3)

def show_prediction_2(image, gt, model, fig_path, start_time):
    batch_size = len(image)
    # mlb = MultiLabelBinarizer()
    # Generate prediction
    prediction = model(image)
    prediction = np.round(prediction, 4)
    # prediction = pd.Series(prediction[0])
    # prediction.index = mlb.classes_
    # prediction = prediction[prediction==1].index.values

    # Dispaly image with prediction    
    fig, axes = plt.subplots(batch_size, 3, figsize=(10,4*batch_size))
    axes[0, 0].set_title('Image')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 2].set_title('Prediction (select >.5)')
    for i in range(batch_size):
        # Display the image
        axes[i, 0].imshow(image[i])
        axes[i, 0].axis('off')

        # Display the ground truth
        axes[i, 1].axis([0, 10, 0, 10])
        axes[i, 1].axis('off')
        axes[i, 1].text(1, 2, '\n'.join(LABELS[np.where(gt[i].numpy() == 1)]), fontsize=12)

        # Display the predictions
        selected = np.where(prediction[i] > (np.max(prediction[i]) * 0.9), '*', ' ')
        combined_array = list(zip(LABELS, prediction[i], selected))
        pred_str = '\n'.join([f"{row[2]} {row[0]}, {row[1]}" for row in combined_array])
        axes[i, 2].axis([0, 10, 0, 10])
        axes[i, 2].axis('off')
        axes[i, 2].text(1, 0, prediction[i], fontsize=10)
        
    # style.use('default')
    filename = os.path.join(fig_path, "sample_predict_" + start_time + ".png")
    print("Saving to", filename)
    plt.savefig(filename)


def _preprocess_val(x, y, info=None):
    x = preprocess_image(
        x, *IMAGE_SIZE,
        is_training=False, color_distort=False, crop='Center')
    return x, y

# load image
customLoader = MIMIC_CXR_JPG_Loader({'train': 0, 'validate': 0, 'test': 2*BATCH_SIZE}, project_folder)
train_tfds, val_tfds, test_tfds = customLoader.load()
val_tfds = val_tfds.shuffle(buffer_size=BATCH_SIZE)
batched_test_tfds = test_tfds.map(_preprocess_val).batch(BATCH_SIZE)

for batch in batched_test_tfds:
    show_prediction_2(*batch, module, os.path.join(project_folder, './out/figs'), START_TIME)
    break
