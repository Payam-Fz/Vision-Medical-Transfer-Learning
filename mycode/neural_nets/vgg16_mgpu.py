#------------------- SET PATH -------------------#

import os
import sys

DEFAULT_JOB_PATH = '/ubc/cs/research/shield/projects/payamfz/medical-ssl-segmentation'
DEFAULT_LOCAL_PATH = '/mnt/samba/research/shield/projects/payamfz/medical-ssl-segmentation'
if os.getcwd().endswith('/job'):
    target_path = DEFAULT_JOB_PATH  # we are running a job
else:
    target_path = DEFAULT_LOCAL_PATH    # we are running locally

code_path = target_path + '/mycode'
if code_path not in sys.path:
  sys.path.append(code_path)

from set_path import get_proj_path, set_path
set_path()

print(sys.path)
print(os.getcwd())

#------------------- IMPORTS -------------------#

import numpy as np
from time import time
from absl import app
from absl import flags

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, VGG16
import tensorboard

from utils.augmentation import preprocess_image
from utils.log import get_curr_datetime, print_time, print_log
from utils.analysis import learning_curves, show_prediction
from loader.mimic_cxr_jpg_loader import MIMIC_CXR_JPG_Loader
from utils.objective_func import soft_f1_loss, f1_score, soft_f1_loss, macro_bce

#------------------- PARAMS -------------------#

FLAGS = flags.FLAGS

_DATASET = flags.DEFINE_string(
    'dataset', 'MIMIC-CXR', '["Chexpert", "MIMIC-CXR", "Noise"]'
)
_OUTPUT_NAME = flags.DEFINE_string(
    'ouput_name', 'unnamed', 'e.g. "finetuned_REMEDIS"'
)
_EPOCHS = flags.DEFINE_integer(
    'epochs', 10, 'Number of epochs to perform fine-tuning.'
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 64, 'Batch size for training.'
)
_IMAGE_SIZE = flags.DEFINE_integer(
    'image_size', 448, 'Input image size.'
)
_TRAIN_SIZE = flags.DEFINE_integer(
    'train_size', 10000, 'Size of training dataset.'
)
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 1e-4, 'Initial learning rate per batch size.'
)
_MOMENTUM = flags.DEFINE_float(
    'momentum', 0.9, 'Momentum parameter.'
)
_WEIGHT_DECAY = flags.DEFINE_float(
    'weight_decay', 1e-6, 'Amount of weight decay to use.'
)
_GPU_MEM_LIMIT = flags.DEFINE_integer(
    'gpu_mem_limit', 11000, 'Limit in megabytes.'
)
_TRANSFER_LEARNING = flags.DEFINE_boolean(
    'transfer_learning', True, 'Whether freeze the original model weights.'
)

#------------------- MODEL -------------------#

class BaseModel(keras.models.Sequential):
    
    def __init__(self, writer, global_batch_size, loss, strategy):
        super().__init__()
        self.writer = writer
        self.my_loss = loss
        self.strategy = strategy
        # self.bce_loss = tf.Variable(float('inf'), trainable=False)
        # self.global_batch_size = global_batch_size
        # with strategy.scope():
        #     self.bce = keras.losses.BinaryCrossentropy(
        #         from_logits=False,
        #         reduction=tf.keras.losses.Reduction.NONE)    # False due to 'sigmoid' activation layer
        #     self.loss = 

    # override train_step to log other metrics
    def train_step(self, data):
        # step training
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.my_loss(y, y_pred)
            # macro_soft_loss = macro_soft_f1(y, y_pred)
            # self.macro_bce_loss.assign(macro_bce(y, y_pred))
            # bce_loss_batch = loss = tf.reduce_sum(self.bce(y, y_pred)) * (1. / self.global_batch_size)
            # self.bce_loss.assign(bce_loss_batch)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
  
        # self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        
        # log metrics
        with self.writer.as_default(step=self._train_counter):
            # for metric in self.compiled_metrics.metrics:
            #     tf.summary.scalar(metric.name, metric.result())
            # tf.summary.scalar('macro_bce_loss', self.macro_bce_loss)
            # tf.summary.scalar('bce_loss', self.bce_loss)
            tf.summary.scalar('macro_soft_f1_loss', loss)
            tf.summary.image('input_image', x, max_outputs=)
        
        # return self.compute_metrics(x, y, y_pred, None)
        return loss
    
    @tf.function
    def distributed_train_step(self, dist_inputs):
        per_replica_losses = self.strategy.run(self.train_step, args=(dist_inputs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

def create_vgg16(input_shape, num_classes, writer, learning_rate, is_transfer_learning, global_batch_size, loss_fn, strategy):
    base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    if is_transfer_learning:
        base_vgg.trainable = False

    model = BaseModel(writer, global_batch_size, loss_fn, strategy)
    model.add(layers.Input(shape=input_shape, name='my_input'))
    for layer in base_vgg.layers:
        model.add(layer)
    model.add(layers.Flatten(name='my_flatten'))
    model.add(layers.Dense(256, activation='relu', name='my_fc_1'))
    model.add(layers.Dense(128, activation='relu', name='my_fc_2'))
    model.add(layers.Dense(num_classes, activation='sigmoid', name='my_output'))
    
    return model

#------------------- MAIN -------------------#

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    
    #------------------- SETUP GPU -------------------#

    GPU_MEM_LIMIT = _GPU_MEM_LIMIT.value
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                # tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEM_LIMIT)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            assert len(logical_gpus) == len(gpus), "Number of logical GPUs were altered"
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            raise
    else:
        assert False, "Running without GPU"

    mirrored_strategy = tf.distribute.MirroredStrategy()
    NUM_GPU = mirrored_strategy.num_replicas_in_sync
    assert NUM_GPU == len(gpus), "Missmatch in number of GPUs"

    #------------------- VARIABLES -------------------#
    
    DATASET = _DATASET.value #@param ["Chexpert", "Camelyon", "MIMIC-CXR", "Noise"]
    BATCH_SIZE = _BATCH_SIZE.value
    TRAIN_SIZE = _TRAIN_SIZE.value
    IMAGE_SIZE = (_IMAGE_SIZE.value, _IMAGE_SIZE.value)
    LEARNING_RATE = _LEARNING_RATE.value
    EPOCHS = _EPOCHS.value
    WEIGHT_DECAY = _WEIGHT_DECAY.value
    CHANNELS = 3

    PROJ_PATH = get_proj_path()
    START_TIME = get_curr_datetime()
    OUTPUT_NAME = _OUTPUT_NAME.value + '_' + START_TIME
    
    TRANSFER_LEARNING = _TRANSFER_LEARNING.value

    #------------------- OUTPUT DIRECTORIES -------------------#
    
    board_dir = os.path.join(PROJ_PATH, 'out', OUTPUT_NAME, 'board')
    figs_dir = os.path.join(PROJ_PATH, 'out', OUTPUT_NAME, 'figs')
    export_dir = os.path.join(PROJ_PATH, 'out', OUTPUT_NAME, 'model')
    checkpoint_dir = os.path.join(export_dir, 'checkpoints', 'epoch-{epoch:02d}_validloss-{val_loss:.2f}.ckpt')
    model_dir = os.path.join(export_dir, 'saved_model')
    
    os.makedirs(board_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(export_dir + '/checkpoints', exist_ok=True)
    os.makedirs(export_dir + '/saved_model', exist_ok=True)

    #------------------- PRINT CONFIGURATION -------------------#

    # Print date and time
    print_log(f'Start: {START_TIME}\n')
    print_log(f'Number of GPUs: {NUM_GPU}')
    print_log(f'Dataset: {DATASET}')
    print_log(f'Training Dataset Size: {TRAIN_SIZE}')
    print_log(f'Epochs: {EPOCHS}')
    print_log(f'Batch size per GPU: {BATCH_SIZE / NUM_GPU}')
    print_log(f'Global Batch size: {BATCH_SIZE}')
    print_log(f'Image size: {IMAGE_SIZE}')

    #------------------- LOAD DATA -------------------#

    with mirrored_strategy.scope():

        if DATASET == 'Noise':
            def generate_fake_tfds_dataset(width, height, channels, num_classes, N=1000, data_type=tf.float32):
                train_examples = np.random.normal(size=[N, width, height, channels])
                classes = np.arange(0, num_classes)
                random_labels = np.random.choice(a=classes, size=N)
                one_hot_encoded = np.zeros((N, num_classes))
                one_hot_encoded[np.arange(N), random_labels] = 1
                train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, one_hot_encoded))
                return train_dataset
            
            num_classes = 14
            train_tfds = generate_fake_tfds_dataset(*IMAGE_SIZE, CHANNELS, num_classes)
        
        elif DATASET == 'Chexpert':
            # TODO: Load chexpert data here.
            num_classes = 14
            raise Exception("not implemented. Please download the chexpert data manually and add code to read here.")

        elif DATASET == 'MIMIC-CXR':
            # customLoader = MIMIC_CXR_JPG_Loader({'train': 360000, 'validate': 2900, 'test': 0}, project_folder)
            data_loader = MIMIC_CXR_JPG_Loader({'train': TRAIN_SIZE, 'validate': 2900, 'test': 0}, PROJ_PATH)
            train_tfds, val_tfds, test_tfds = data_loader.load(['has_label', 'frontal_view', 'unambiguous_label'])
            num_classes = data_loader.info()['num_classes']
            for key, value in data_loader.info().items():
                print_log(f'{key}: {value}')

        else:
            raise Exception('The Data Type specified does not have data loading defined.')

        
        # Chexpert: TFDS.has Supervised - produces binary labels the way we were using them
        #           Chexpert data loader fails unless you have it downloaded - download & put in this directory
        #           Have self-selectors
        # Noise:    fake tfds for testing
        @tf.function
        def _preprocess_train(x, y, info=None):
            x = preprocess_image(x, *IMAGE_SIZE,
                is_training=True, color_distort=False, crop='Center')
            x = tf.ensure_shape(x, [*IMAGE_SIZE,CHANNELS])
            y = tf.ensure_shape(y, [num_classes])
            return x, y

        @tf.function
        def _preprocess_val(x, y, info=None):
            x = preprocess_image(x, *IMAGE_SIZE,
                is_training=False, color_distort=False, crop='Center')
            x = tf.ensure_shape(x, [*IMAGE_SIZE,CHANNELS])
            y = tf.ensure_shape(y, [num_classes])
            return x, y

        train_tfds = train_tfds.shuffle(buffer_size=2*BATCH_SIZE)
        batched_train_tfds = train_tfds.map(_preprocess_train).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_tfds = val_tfds.shuffle(buffer_size=2*BATCH_SIZE)
        batched_val_tfds = val_tfds.map(_preprocess_val).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

        # TODO: in case improves performance
        # AUTOTUNE = tf.data.experimental.AUTOTUNE
        # batched_train_tfds = batched_train_tfds.prefetch(buffer_size=AUTOTUNE)

        # next_batch = tf.data.make_one_shot_iterator(batched_train_tfds).get_next()

        for f, l in batched_train_tfds.take(1):
            print_log("Shape of image batch:", f.shape.as_list())
            print_log("Shape of labels batch:", l.shape.as_list())

        #------------------- SETUP TRAINING -------------------#

        # Define the Keras TensorBoard callback.
        summary_writer = tf.summary.create_file_writer(board_dir)
        tf.summary.trace_on(graph=True)

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=board_dir,
            update_freq=20,   # log every 'update_freq' batch
            profile_batch=(5,10))  # do the profiling for this range of batch
        earlystopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            patience=5)
        checkpointer_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            save_weights_only=True,
            monitor='val_loss',
            verbose=1,
            save_best_only=True)
        
        #------------------- CREATE MODEL -------------------#

        def compute_loss(labels, predictions):
            per_example_loss = soft_f1_loss(labels, predictions)
            loss = tf.nn.compute_average_loss(per_example_loss)
            return loss

        model = create_vgg16(
            input_shape=(*IMAGE_SIZE,CHANNELS),
            num_classes=num_classes,
            writer=summary_writer,
            learning_rate=LEARNING_RATE,
            is_transfer_learning=TRANSFER_LEARNING,
            global_batch_size=BATCH_SIZE,
            loss_fn=compute_loss,
            strategy=mirrored_strategy)

            
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        bce = keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        
        model.compile(
            loss=compute_loss,
            optimizer=optimizer,
            metrics=[bce])

        print(model.summary())


        #------------------- PERFORM TRAINING -------------------#

        start = time()
        history = model.fit(
            batched_train_tfds,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=batched_val_tfds,
            callbacks=[tensorboard_callback, earlystopping_callback, checkpointer_callback])
        
        print_log('history:')
        for key, value in history.history.items():
                print_log(f'{key}: {value}')


        #------------------- RESULTS -------------------#
        
        print_log('\nTraining took {}'.format(print_time(time()-start)))
        
        with summary_writer.as_default():
            tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=board_dir)

        losses, val_losses, macro_f1s, val_macro_f1s = learning_curves(history, figs_dir)

        print_log("Macro soft-F1 loss: %.2f" %val_losses[-1])
        print_log("Macro F1-score: %.2f" %val_macro_f1s[-1])

        for batch in batched_val_tfds:
            show_prediction(*batch, model, figs_dir)
            break
    
    
        # #------------------- SAVE MODEL -------------------#
        model.save(model_dir, save_format='tf')
        print_log(f"Model was exported to this path: '{model_dir}'")
        
        print_log('\nDONE!')

if __name__ == '__main__':
    app.run(main)