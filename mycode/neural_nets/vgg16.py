#------------------- SETUP -------------------#

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

from utils.augmentation import preprocess_image
from utils.log import get_curr_datetime, print_time, print_log
from utils.analysis import learning_curves, show_prediction, log_eval_metrics
from loader.mimic_cxr_jpg_loader import MIMIC_CXR_JPG_Loader
from utils.objective_func import soft_f1_loss, macro_f1_score, macro_bce, global_accuracy, global_precision, global_recall

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
    'learning_rate', 0.1, 'Initial learning rate per batch size.'
)
_MOMENTUM = flags.DEFINE_float(
    'momentum', 0.9, 'Momentum parameter.'
)
_WEIGHT_DECAY = flags.DEFINE_float(
    'weight_decay', 1e-6, 'Amount of weight decay to use.'
)
# _GPU_MEM_LIMIT = flags.DEFINE_integer(
#     'gpu_mem_limit', 11000, 'Limit in megabytes.'
# )
_LOAD_CHECKPOINT = flags.DEFINE_string(
    'load_checkpoint', None, 'Address of checkpoint to be used. None if no checkpoint.'
)
_MODE = flags.DEFINE_string(
    'mode', 'train_then_eval', '["train_then_eval", "eval" ]'
)

_MIN_UNFREEZE_BLOCKS = flags.DEFINE_integer(
    'min_unfreeze_blocks', 0, 'Will unfreeze this number of blocks before starting the training.'
)
_MAX_UNFREEZE_BLOCKS = flags.DEFINE_integer(
    'max_unfreeze_blocks', 0, 'Will unfreeze this number of blocks by the end of the training.'
)
# _GRADUAL_UNFREEZE = flags.DEFINE_boolean(
#     'gradual_unfreeze', False, 'Will unfreeze blocks one after the other. This will overrite the "unfreeze_blocks" argument.'
# )

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        # for gpu in gpus:
        #     # tf.config.experimental.set_memory_growth(gpu, True)
        #     tf.config.set_logical_device_configuration(
        #         gpu,
        #         [tf.config.LogicalDeviceConfiguration(memory_limit=10400)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class BaseModel(keras.models.Sequential):
    
    def __init__(self):
        super().__init__()
        self.writer = None
        
    def set_writer(self, writer):
        self.writer = writer

    # override train_step to log other metrics
    def train_step(self, data):
        # step training
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        
        # log metrics
        with self.writer.as_default(step=self._train_counter):
            for metric in self.compiled_metrics.metrics:
                tf.summary.scalar(metric.name, metric.result())
            tf.summary.scalar('loss', loss)
            tf.summary.image('input_image', x, max_outputs=10)
        
        return self.compute_metrics(x, y, y_pred, None)

def create_vgg16(input_shape, num_classes):
    base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_vgg.trainable = False

    model = BaseModel()
    model.add(layers.Input(shape=input_shape, name='my_input'))
    for layer in base_vgg.layers:
        model.add(layer)
    model.add(layers.Flatten(name='my_flatten'))
    model.add(layers.Dense(256, activation='relu', name='my_fc_1'))
    model.add(layers.Dense(128, activation='relu', name='my_fc_2'))
    model.add(layers.Dense(num_classes, activation='sigmoid', name='my_output'))
    
    return model

def unfreeze_layers(model, total_blocks_count, unfreeze_blocks_count):
    for layer in model.layers:
        layer_name = layer.name
        if layer_name.startswith('block'):
            block_number = int(layer_name.split('_')[0].replace('block', ''))
            if unfreeze_blocks_count >= total_blocks_count - block_number + 1:
                layer.trainable = True

def perform_training(model, train_tfds, val_tfds, callbacks, epochs, batch_size):
    start = time()
    history = model.fit(
        train_tfds,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=val_tfds,
        callbacks=callbacks)

    duration = time()-start
    return duration, history
    
def log_training_progress(duration, history, figs_dir):
    print_log(f'Training took {print_time(duration)}')
    print_log('history of metrics:')
    for key, value in history.history.items():
        print_log(f'\t{key}: {value}')
        
    losses, val_losses, macro_f1s, val_macro_f1s = learning_curves(history, figs_dir)
    print_log('Validation loss: %.4f' %val_losses[-1])
    print_log('Validation Macro F1-score: %.4f' %val_macro_f1s[-1])

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    
    #------------------- VARIABLES -------------------#
    
    LOAD_CHECKPOINT = _LOAD_CHECKPOINT.value
    MODE = _MODE.value
    assert MODE in ["train_then_eval", "eval"], 'Invalid Mode argument'
    
    DATASET = _DATASET.value
    assert DATASET in ["Chexpert", "MIMIC-CXR", "Noise"], 'Invalid Dataset argument'
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
    
    TOTAL_CONV_BLOCKS = 5
    MIN_UNFREEZE_BLOCKS = _MIN_UNFREEZE_BLOCKS.value
    MAX_UNFREEZE_BLOCKS = min(_MAX_UNFREEZE_BLOCKS.value, TOTAL_CONV_BLOCKS)
    assert MIN_UNFREEZE_BLOCKS <= MAX_UNFREEZE_BLOCKS, 'Invalid Min and Max Unfrozen blocks specified'
    
    learning_rate_schedule = {
        0: 1e-3,
        1: 1e-5,
        2: 1e-5,
        3: 1e-6,
        4: 1e-6,
        5: 1e-6
    }
    

    #------------------- PRINT CONFIGURATION -------------------#

    print_log('\n------------------ Configuration ------------------')
    print_log(f'Start: {START_TIME}')
    print_log(f'\nMode: {MODE}')
    print_log(f'Unfreeze blocks: start {MIN_UNFREEZE_BLOCKS}, end {MAX_UNFREEZE_BLOCKS}')
    print_log(f'Continue from checkpoint: {False if LOAD_CHECKPOINT is None else LOAD_CHECKPOINT}')
    print_log(f'\nDataset: {DATASET}')
    print_log(f'Training Dataset Size: {TRAIN_SIZE}')
    print_log(f'Batch size: {BATCH_SIZE}')
    print_log(f'Image size: {IMAGE_SIZE}')
    print_log(f'Max Epochs per training round: {EPOCHS}')
    print_log(f'Default Learning Rate: {LEARNING_RATE}')
    print_log(f'\nGPUs: {tf.config.list_physical_devices("GPU")}')
    print_log(f'CPUs: {tf.config.list_physical_devices("CPU")}')

    #------------------- LOAD DATA -------------------#
    print_log('\n------------------ Data ------------------')
    
    # Chexpert: TFDS.has Supervised - produces binary labels the way we were using them
    #           Chexpert data loader fails unless you have it downloaded - download & put in this directory
    #           Have self-selectors
    # Noise:    fake tfds for testing
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
        # max size: {'train': 360000, 'validate': 2900, 'test': 0}
        data_loader = MIMIC_CXR_JPG_Loader({'train': TRAIN_SIZE, 'validate': 2900, 'test': 512}, PROJ_PATH)
        train_tfds, val_tfds, test_tfds = data_loader.load(['has_label', 'frontal_view', 'unambiguous_label'])
        num_classes = data_loader.info()['num_classes']
        
        for key, value in data_loader.info().items():
            print_log(f'{key}: {value}')

    else:
        raise Exception('The Data Type specified does not have data loading defined.')


    @tf.function
    def _preprocess_train(x, y, info=None):
        x = preprocess_image(x, *IMAGE_SIZE,
            is_training=True, color_distort=False, crop='Center')
        x = tf.ensure_shape(x, [*IMAGE_SIZE,3])
        y = tf.ensure_shape(y, [num_classes])
        return x, y

    @tf.function
    def _preprocess_val(x, y, info=None):
        x = preprocess_image(x, *IMAGE_SIZE,
            is_training=False, color_distort=False, crop='Center')
        x = tf.ensure_shape(x, [*IMAGE_SIZE,3])
        y = tf.ensure_shape(y, [num_classes])
        return x, y

    train_tfds = train_tfds.map(_preprocess_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_tfds = train_tfds.shuffle(buffer_size=2*BATCH_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_tfds = val_tfds.map(_preprocess_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_tfds = val_tfds.shuffle(buffer_size=2*BATCH_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_tfds = test_tfds.map(_preprocess_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for f, l in train_tfds.take(1):
        print_log("Shape of image batch:", f.shape.as_list())
        print_log("Shape of labels batch:", l.shape.as_list())

    
    #------------------- CREATE MODEL -------------------#
    
    model = create_vgg16(input_shape=(*IMAGE_SIZE,CHANNELS), num_classes=num_classes)
    print(model.summary())
    
    is_first_round = True
    last_checkpoint_dir = ''
    tf.summary.trace_on(graph=True)
    main_out_dir = os.path.join(PROJ_PATH, 'out', OUTPUT_NAME)
    os.makedirs(main_out_dir, exist_ok=True)
    
    if MODE == 'eval':
        round_range = range(1)
    else:
         round_range = range(MIN_UNFREEZE_BLOCKS, MAX_UNFREEZE_BLOCKS + 1)
    
    # This loop will always happen at least once
    for num_unfrozen_blocks in round_range:
        print_log('\n------------------ Setup Round ------------------')
        
        #------------------- OUTPUT DIRECTORIES -------------------#
        
        round_dir = os.path.join(main_out_dir, f'{num_unfrozen_blocks}_unfrozen_block')
        board_dir = os.path.join(round_dir, 'board')
        figs_dir = os.path.join(round_dir, 'figs')
        export_dir = os.path.join(round_dir, 'model')
        checkpoint_dir = os.path.join(export_dir, 'checkpoints')
        model_dir = os.path.join(export_dir, 'saved_model')
        
        os.makedirs(board_dir, exist_ok=True)
        os.makedirs(figs_dir, exist_ok=True)
        os.makedirs(export_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        
        #------------------- SETUP METRICS AND DRIVERS -------------------#

        summary_writer = tf.summary.create_file_writer(board_dir)
        model.set_writer(summary_writer)

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=board_dir,
            update_freq=20,   # log metrics every this many batches
            profile_batch=(24,40))  # do the profiling for this range of batch
        earlystopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            min_delta=0.001,   # how much change is considered an improvement
            patience=2,
            start_from_epoch=0)
        checkpointer_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'epoch-{epoch:02d}_valloss-{val_loss:.4f}.ckpt'),
            save_weights_only=True,
            monitor='val_loss',
            verbose=1,
            save_best_only=True) # so that the next round loads the best weights
        callbacks = [tensorboard_callback, earlystopping_callback, checkpointer_callback]
        
        learning_rate = learning_rate_schedule.get(num_unfrozen_blocks, LEARNING_RATE)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        bce = keras.losses.BinaryCrossentropy(from_logits=False)
        auc = keras.metrics.AUC(
            curve='ROC',
            name='AUC',
            multi_label=True,
            num_labels=num_classes,
            from_logits=False)
        metrics = [macro_f1_score, soft_f1_loss, auc, global_accuracy, global_precision, global_recall]
        
        
        #------------------- MODIFY MODEL -------------------#
        
        if MODE != 'eval':
            # Unfreeze blocks (if needed)
            if num_unfrozen_blocks > 0:
                unfreeze_layers(model, TOTAL_CONV_BLOCKS, num_unfrozen_blocks)
            print_log(f'Unfreezing {num_unfrozen_blocks} blocks...')
            print_log(f'Total trainable weights: {len(model.trainable_weights)}')
            for weight in model.trainable_weights:
                print_log(f'\t{weight.name}')
            print_log(f'Learning rate = {learning_rate}')
        
        # (Re)compile model
        # NOTE: compiling will reset weights to random
        model.compile(loss=bce, optimizer=optimizer, metrics=metrics)
        
        # Load weights
        if is_first_round:
            if LOAD_CHECKPOINT:
                load_checkpoint_dir = os.path.join(PROJ_PATH, LOAD_CHECKPOINT)
                latest = tf.train.latest_checkpoint(load_checkpoint_dir)
                model.load_weights(latest)
                print_log(f'Loading weights from {latest}')
        else:
            # Reload the latest weights from previous raound
            latest = tf.train.latest_checkpoint(last_checkpoint_dir)
            model.load_weights(latest)
            print_log(f'Loading weights from {latest}')
        
        if MODE != 'eval':
            
            #------------------- PERFORM TRAINING -------------------#
            
            print_log('\n------------------ Training ------------------')
            # Train with newly unfrozen blocks
            duration, history = perform_training(model, train_tfds, val_tfds, callbacks, EPOCHS, BATCH_SIZE)
            print_log('\n------------------ Training Round Summary ------------------')
            log_training_progress(duration, history, figs_dir)
            
        
            #------------------- SAVE MODEL -------------------#
        
            model.save(model_dir, save_format='tf')
            print_log(f'Saved model to: "{model_dir}"')
            
            # with summary_writer.as_default():
            #     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=board_dir)
        
        
        is_first_round = False
        last_checkpoint_dir = checkpoint_dir

    
    #------------------- EVALUATE -------------------#
    
    print_log('\n------------------ Evaluate ------------------')
    
    del train_tfds
    del val_tfds
    test_tfds = test_tfds.shuffle(buffer_size=10*BATCH_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    for f, l in test_tfds.take(1):
        show_prediction(f, l, model, main_out_dir)

    log_eval_metrics(
        dataset=test_tfds,
        model=model,
        metrics=[bce, auc, macro_f1_score, soft_f1_loss, global_accuracy, global_precision, global_recall])
    
    print_log('\nDONE!')

if __name__ == '__main__':
    app.run(main)