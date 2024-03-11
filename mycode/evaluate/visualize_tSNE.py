# Credit: https://stackoverflow.com/questions/71500106/how-to-implement-t-sne-in-tensorflow

import os
import tensorflow as tf
import pathlib
from sklearn.manifold import TSNE
import numpy as np
from  matplotlib import pyplot as plt
from loader.mimic_cxr_jpg_loader import MIMIC_CXR_JPG_Loader
from set_path import get_proj_path
from utils.augmentation import preprocess_image

def show_tSNE(model, batch_size=32, num_batches=5, signature_name=None):
    # ------------- Variables ------------
    input_tensor_shape = (448,448)  # default
    
    # ------------- Process model -----------
    if hasattr(model, 'signatures'):
        input_signature_name = signature_name if signature_name is not None else list(model.signatures.keys())[0]
        input_signature = model.signatures[input_signature_name]
        provided_shape = input_signature.inputs[0].shape[1:3]    # Shape is like (None, 448, 448, 3)
        if None not in provided_shape:
            input_tensor_shape = provided_shape

    inp_height, inp_width = input_tensor_shape[0], input_tensor_shape[1]
    print(f'input size (width, height): ({inp_width}, {inp_height})')
    
    # ------------- load MIMIC-CXR dataset -------------
    def _preprocess_val(x, y, info=None):
        x = preprocess_image(
            x, inp_width, inp_height,
            is_training=False, color_distort=False, crop='Center')
        return x, y
    
    data_loader = MIMIC_CXR_JPG_Loader({'train': batch_size*num_batches*5, 'validate': 0, 'test': 0}, get_proj_path())
    train_tfds, _, _ = data_loader.load(['has_label', 'frontal_view'])
    class_names = data_loader.metadata['class_names']

    train_tfds = train_tfds.shuffle(buffer_size=batch_size*2)
    batched_train_tfds = train_tfds.map(_preprocess_val).batch(batch_size)
    samples = list(batched_train_tfds.take(num_batches))
    test_input = tf.concat([x for x, y in samples], axis=0)
    test_labels = np.concatenate([y for x, y in samples], axis=0)
    
    print('input shape:', test_input.shape)
    print('labels shape:', test_labels.shape)
    print('class_names:',class_names)
    
    plt.imshow(test_input[0], cmap='gray')
    plt.axis('off')
    plt.show()
    print('label:',test_labels[0])
    

    # ------------- inference -------------

    infer = model(test_input)
    if isinstance(infer, dict):
        print('Output signatures:', infer.keys())
        input_signature_name = signature_name if signature_name is not None else list(infer.keys())[0]
        features = infer[input_signature_name]
    else:
        features = infer.numpy()
    
    print("Features shape:",features.shape)
    
    # pred_labels = np.argmax(model(test_input), axis=-1)
    tsne = TSNE(n_components=2).fit_transform(features)

    def scale_to_01_range(x):
        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    colors = ['red', 'cyan', 'green', 'pink', 'orange', 'purple', 'blue', 'yellow', 'gray']
    assert len(colors) >= len(class_names), "not enough colors"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # for every class, we'll add a scatter plot separately
    for class_idx, label in enumerate(class_names):
        # find the samples of the current class in the data
        color = colors[class_idx]
        print('processing class:', label, color)
        sample_idxs = [sample_idx for sample_idx, sample_lbl in enumerate(test_labels) if sample_lbl[class_idx] == 1]
        current_tx = np.take(tx, sample_idxs)
        current_ty = np.take(ty, sample_idxs)
        ax.scatter(current_tx, current_ty, c=color, label=label)

    ax.legend(loc='best')
    return fig
    