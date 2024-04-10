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
import tensorflow_hub as hub

from utils.augmentation import preprocess_image
from utils.log import get_curr_datetime, print_time, print_log
from utils.analysis import learning_curves, show_prediction, log_eval_metrics
from loader.mimic_cxr_jpg_loader import MIMIC_CXR_JPG_Loader
from utils.objective_func import soft_f1_loss, macro_f1_score, macro_bce, global_accuracy, global_precision, global_recall
# from neural_nets.components.patch_encoder import PatchEncoder
# from neural_nets.components.patch_creation import Patches
from loader.vit_loader import load_model

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

class BaseModel(keras.Model):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.relu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def transformer_block(encoded_patches, projection_dim, num_heads, mlp_units):
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Create a multi-head attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.1
    )(x1, x1)
    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])
    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    # MLP.
    x3 = mlp(x3, hidden_units=mlp_units, dropout_rate=0.1)
    # Skip connection 2.
    out = layers.Add()([x3, x2])
    return out

def create_vit(input_shape, num_classes):
    base_vit = hub.KerasLayer(
        "https://www.kaggle.com/models/spsayakpaul/vision-transformer/TensorFlow2/vit-b16-fe/1",
        trainable=False,
    )
    base_vit.trainable = False

    inputs = keras.Input(input_shape, name='my_input')
    base_output = base_vit(inputs)
    x = layers.Dense(256, activation='relu', name='my_fc_1')(base_output)
    x = layers.Dense(128, activation='relu', name='my_fc_2')(x)
    x = layers.Dense(num_classes, activation='sigmoid', name='my_output')(x)
    model = BaseModel(inputs=inputs, outputs=x, name='my_vit')
    
    return model

def unfreeze_layers(model, unfreeze_blocks_count):
    # For ViT, unfreeze the whole model
    if (unfreeze_blocks_count > 0):
        for layer in model.layers:
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
    
    TOTAL_CONV_BLOCKS = 1
    MIN_UNFREEZE_BLOCKS = _MIN_UNFREEZE_BLOCKS.value
    MAX_UNFREEZE_BLOCKS = min(_MAX_UNFREEZE_BLOCKS.value, TOTAL_CONV_BLOCKS)
    assert MIN_UNFREEZE_BLOCKS <= MAX_UNFREEZE_BLOCKS, 'Invalid Min and Max Unfrozen blocks specified'
    
    ES_PATIENCE = 3
    learning_rate_schedule = {
        0: 1e-3,
        1: 1e-6
    }

    # ----------- Model Configs -----------
    assert IMAGE_SIZE == (224, 224), "Wrong image size for this model"
    PATCH_SIZE = 16 # Results in 14 x 14 patches
    model_path = 'base-models/ViT/vit_b16_patch16_224-i1k_pretrained.zip'

    

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
    print_log(f'Learning Rate: {learning_rate_schedule}')
    print_log(f'Early Stopping Patience: {ES_PATIENCE}')
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


    # crop_layer = keras.layers.CenterCrop(*IMAGE_SIZE)
    # norm_layer = keras.layers.Normalization(
    #     mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    #     variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2],
    # )
    # rescale_layer = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1)


    # def preprocess_vit_image(image, model_type, size=IMAGE_SIZE[0]):
    #     # Turn the image into a numpy array and add batch dim.
    #     image = np.array(image)
    #     image = ops.expand_dims(image, 0)

    #     # If model type is vit rescale the image to [-1, 1].
    #     if model_type == "original_vit":
    #         image = rescale_layer(image)

    #     # Resize the image using bicubic interpolation.
    #     resize_size = int((256 / 224) * size)
    #     image = ops.image.resize(image, (resize_size, resize_size), interpolation="bicubic")

    #     # Crop the image.
    #     image = crop_layer(image)

    #     # If model type is DeiT or DINO normalize the image.
    #     if model_type != "original_vit":
    #         image = norm_layer(image)

    #     return ops.convert_to_numpy(image)

    # @tf.function
    # def _preprocess_train(x, y, info=None):
    #     x = preprocess_image(x, *IMAGE_SIZE, is_training=True, probability=0.4, v2=False)
    #     x = tf.image.convert_image_dtype(x, dtype=tf.uint8)
    #     x = tf.keras.applications.imagenet_utils.preprocess_input(x)
    #     x = tf.ensure_shape(x, [*IMAGE_SIZE,3])
    #     y = tf.ensure_shape(y, [num_classes])
    #     return x, y

    def _preprocess_val(x, y, info=None):
        x = preprocess_image(x, *IMAGE_SIZE, is_training=False)
        x = tf.image.convert_image_dtype(x, dtype=tf.uint8) # scales image to 0-255 # in range
        # x = tf.multiply(x, 255.0)   # still in range
        # x = tf.keras.applications.resnet50.preprocess_input(x) # keeps type and range correct
        # x = tf.cast(x, tf.float32)  # only converts type without scaling
        x = tf.keras.applications.imagenet_utils.preprocess_input(x)  # scales to 0-1 float32 # goes off if float is passed
        # x = tf.divide(x, 255.0)
        
        
        x = tf.image.convert_image_dtype(x, dtype=tf.float32) # scales image to 0.0-1.0 # in range
        x = tf.subtract(tf.multiply(x, 2.0), 1.0)
        
        
        # x = tf.clip_by_value(x, -1.0, 1.0)
        x = tf.ensure_shape(x, [*IMAGE_SIZE,3])
        y = tf.ensure_shape(y, [num_classes])
        return x, y

    for f, l, info in train_tfds.take(1):
        print_log('max value before preprocess',tf.reduce_max(f))
        print_log('min value before preprocess',tf.reduce_min(f))
        print_log("Shape of image batch:", f.shape.as_list())
        print_log("Shape of labels batch:", l.shape.as_list())
    
    
    train_tfds = train_tfds.map(_preprocess_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_tfds = train_tfds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    for f, l in train_tfds.take(1):
        print_log('max value after preprocess',tf.reduce_max(f))
        print_log('min value after preprocess',tf.reduce_min(f))
        print_log("Shape of image batch:", f.shape.as_list())
        print_log("Shape of labels batch:", l.shape.as_list())

    
    #------------------- CREATE MODEL -------------------#
    
    model = create_vit((*IMAGE_SIZE,CHANNELS), num_classes)
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
            )  # do the profiling for this range of batch
        earlystopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            # min_delta=0.001,   # how much change is considered an improvement
            patience=ES_PATIENCE,
            start_from_epoch=0)
        checkpointer_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'epoch-{epoch:02d}_valloss-{val_loss:.4f}.ckpt'),
            save_weights_only=True,
            monitor='val_loss',
            verbose=1,
            save_best_only=True) # so that the next round loads the best weights
        callbacks = [tensorboard_callback]
        
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
                unfreeze_layers(model, num_unfrozen_blocks)
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
            duration, history = perform_training(model, train_tfds, train_tfds, callbacks, EPOCHS, BATCH_SIZE)
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
    
    for f, l in train_tfds.take(1):
        show_prediction(f, l, model, main_out_dir)

    log_eval_metrics(
        dataset=train_tfds,
        model=model,
        metrics=[bce, auc, macro_f1_score, soft_f1_loss, global_accuracy, global_precision, global_recall])
    
    print_log('\nDONE!')

if __name__ == '__main__':
    app.run(main)