import os
from absl import app
from absl import flags

import tensorflow as tf
import tensorflow_hub as hub
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np

from loader.mimic_cxr_jpg_loader import MIMIC_CXR_JPG_Loader
from utils.augmentation import preprocess_image
from utils.analysis import *

from ..constants import PROJECT_FOLDER

FLAGS = flags.FLAGS

_MODEL_NAME = flags.DEFINE_string(
    'model_name', 'unnamed', 'e.g. "finetuned_REMEDIS"'
)
_MODEL_PATH = flags.DEFINE_string(
    'model_path', './base-models/simclr/r50_2x_sk0/hub/', 'e.g. "./base-models/simclr/r152_2x_sk1/hub/" or "./base-models/remedis/cxr-152x2-remedis-m/"'
)
_IMAGE_SIZE = flags.DEFINE_integer(
    'image_size', 448, 'Input image size.'
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 64, 'Batch size for training.'
)


# def show_prediction_2(image, gt, model, fig_path, start_time):
#     batch_size = len(image)
#     # mlb = MultiLabelBinarizer()
#     # Generate prediction
#     prediction = model(image)
#     prediction = np.round(prediction, 4)
#     # prediction = pd.Series(prediction[0])
#     # prediction.index = mlb.classes_
#     # prediction = prediction[prediction==1].index.values

#     # Dispaly image with prediction    
#     fig, axes = plt.subplots(batch_size, 3, figsize=(10,4*batch_size))
#     axes[0, 0].set_title('Image')
#     axes[0, 1].set_title('Ground Truth')
#     axes[0, 2].set_title('Prediction (select >.5)')
#     for i in range(batch_size):
#         # Display the image
#         axes[i, 0].imshow(image[i])
#         axes[i, 0].axis('off')

#         # Display the ground truth
#         axes[i, 1].axis([0, 10, 0, 10])
#         axes[i, 1].axis('off')
#         axes[i, 1].text(1, 2, '\n'.join(LABELS[np.where(gt[i].numpy() == 1)]), fontsize=12)

#         # Display the predictions
#         selected = np.where(prediction[i] > (np.max(prediction[i]) * 0.9), '*', ' ')
#         combined_array = list(zip(LABELS, prediction[i], selected))
#         pred_str = '\n'.join([f"{row[2]} {row[0]}, {row[1]}" for row in combined_array])
#         axes[i, 2].axis([0, 10, 0, 10])
#         axes[i, 2].axis('off')
#         axes[i, 2].text(1, 0, prediction[i], fontsize=10)
            
#         # style.use('default')
#         filename = os.path.join(fig_path, "sample_predict_" + start_time + ".png")
#         print("Saving to", filename)
#         plt.savefig(filename)
  

def main(argv):
    
    #_____________SETUP_____________
    MODEL_NAME = _MODEL_NAME.value
    MODEL_PATH = _MODEL_PATH.value
    IMAGE_SIZE = (_IMAGE_SIZE.value, _IMAGE_SIZE.value)
    BATCH_SIZE = _BATCH_SIZE.value
    
    START_TIME = get_curr_datetime()

    num_images = 5*BATCH_SIZE

    #_____________LOAD MODEL______________
    model_path = os.path.join(project_folder, MODEL_PATH)
    
    # model = hub.load(model_url)
    if 'remedis' in MODEL_NAME:
        model = tf.saved_model.load(model_path, tags=['serve'])
    elif 'simclr' in MODEL_NAME:
        model = tf.saved_model.load(model_path, tags=[]).signatures['default']
    else:
        assert False, 'seems like model is not supported'
    
    #_____________ TENSORBOARD LOGGING ____________
    logdir=os.path.join(project_folder, "out/board", "evaluate_" + MODEL_NAME + "_" + START_TIME)
    summary_writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)
    # tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")


    #___________LOAD IMAGE__________
    
    # Chest X-Ray: The image is of shape (<BATCH_SIZE>, 448, 448, 3)
    
    def _preprocess_val(x, y, info=None):
        x = preprocess_image(
            x, *IMAGE_SIZE,
            is_training=False, color_distort=False, crop='Center')
        return x, y

    custom_loader = MIMIC_CXR_JPG_Loader({'train': 0, 'validate': 0, 'test': num_images}, project_folder)
    _, _, test_tfds = custom_loader.load()
    test_tfds = test_tfds.shuffle(buffer_size=BATCH_SIZE)
    batched_test_tfds = test_tfds.map(_preprocess_val).batch(BATCH_SIZE)

    #_____________ RUN MODEL & SCORE ____________
    
    @tf.function
    def eval_step(x):
        return model(x)
    
    # Main loop for inference and logging
    counter = 0
    for batch in batched_test_tfds:
        images, labels = batch
        images = images.numpy()
        labels = labels.numpy()
        print('________images.shape:', images.shape)
        print('________images:', images)
        print('________labels.shape:', labels.shape)
        print('________labels:', labels)
        # images = tf.convert_to_tensor(list(images), dtype=tf.float32)

        # Perform inference using the loaded model
        predictions = eval_step(images)
        print('predictions raw:', predictions)
        predictions = predictions.numpy()
        print('________predictions.shape:', predictions.shape)
        print('predictions:', predictions)

        # Calculate the classification loss
        loss = macro_soft_f1(labels, predictions)

        # Log the loss to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('macro_soft_f1 loss', loss, step=counter)
        
        counter = counter + 1
    
    summary_writer.close()
    
    #_____________ SHOW REULTS FOR 1 BATCH ____________
    for batch in batched_test_tfds:
        show_prediction(*batch, model, os.path.join(project_folder, './out/figs'), MODEL_NAME + START_TIME)
        break
    
    print('\nDONE!')

if __name__ == '__main__':
    app.run(main)
