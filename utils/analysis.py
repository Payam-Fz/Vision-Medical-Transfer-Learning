# MIT License
#
# Copyright (c) 2019 Mohamed-Achref MAIZA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
# ==============================================================================
# Credit:
# multi-label-soft-f1 repository
# by Mohamed-Achref Maiza
# Accessed 2024-01-25
# https://github.com/ashrefm/multi-label-soft-f1/blob/master/utils.py

import os
# import shutil
# import urllib.error
# import urllib.request

import matplotlib.pyplot as plt
import matplotlib.style as style

import numpy as np
import pandas as pd
import tensorflow as tf

from time import time
from datetime import datetime
# from sklearn.preprocessing import MultiLabelBinarizer

from loader.labels import LABELS


def learning_curves(history, fig_path, start_time):
    """Plot the learning curves of loss and macro f1 score 
    for the training and validation datasets.
    
    Args:
        history: history callback of fitting a tensorflow keras model 
    """
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    macro_f1 = history.history['macro_f1']
    val_macro_f1 = history.history['val_macro_f1']
    
    epochs = len(loss)

    style.use("bmh")
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs+1), loss, label='Training Loss')
    plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs+1), macro_f1, label='Training Macro F1-score')
    plt.plot(range(1, epochs+1), val_macro_f1, label='Validation Macro F1-score')
    plt.legend(loc='lower right')
    plt.ylabel('Macro F1-score')
    plt.title('Training and Validation Macro F1-score')
    plt.xlabel('epoch')

    filename = os.path.join(fig_path, "learning_curve_" + start_time + ".png")
    print("Saving to", filename)
    plt.savefig(filename)
    
    return loss, val_loss, macro_f1, val_macro_f1


def perf_grid(ds, target, label_names, model, n_thresh=100):
    """Computes the performance table containing target, label names,
    label frequencies, thresholds between 0 and 1, number of tp, fp, fn,
    precision, recall and f-score metrics for each label.
    
    Args:
        ds (tf.data.Datatset): contains the features array
        target (numpy array): target matrix of shape (BATCH_SIZE, N_LABELS)
        label_names (list of strings): column names in target matrix
        model (tensorflow keras model): model to use for prediction
        n_thresh (int) : number of thresholds to try
        
    Returns:
        grid (Pandas dataframe): performance table 
    """
    
    # Get predictions
    y_hat_val = model.predict(ds)
    # Define target matrix
    y_val = target
    # Find label frequencies in the validation set
    label_freq = target.sum(axis=0)
    # Get label indexes
    label_index = [i for i in range(len(label_names))]
    # Define thresholds
    thresholds = np.linspace(0,1,n_thresh+1).astype(np.float32)
    
    # Compute all metrics for all labels
    ids, labels, freqs, tps, fps, fns, precisions, recalls, f1s = [], [], [], [], [], [], [], [], []
    for l in label_index:
        for thresh in thresholds:   
            ids.append(l)
            labels.append(label_names[l])
            freqs.append(round(label_freq[l]/len(y_val),2))
            y_hat = y_hat_val[:,l]
            y = y_val[:,l]
            y_pred = y_hat > thresh
            tp = np.count_nonzero(y_pred  * y)
            fp = np.count_nonzero(y_pred * (1-y))
            fn = np.count_nonzero((1-y_pred) * y)
            precision = tp / (tp + fp + 1e-16)
            recall = tp / (tp + fn + 1e-16)
            f1 = 2*tp / (2*tp + fn + fp + 1e-16)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            
    # Create the performance dataframe
    grid = pd.DataFrame({
        'id':ids,
        'label':labels,
        'freq':freqs,
        'threshold':list(thresholds)*len(label_index),
        'tp':tps,
        'fp':fps,
        'fn':fns,
        'precision':precisions,
        'recall':recalls,
        'f1':f1s})
    
    grid = grid[['id', 'label', 'freq', 'threshold',
                 'tp', 'fn', 'fp', 'precision', 'recall', 'f1']]
    
    return grid


def print_time(t):
    """Function that converts time period in seconds into %h:%m:%s expression.
    Args:
        t (int): time period in seconds
    Returns:
        s (string): time period formatted
    """
    h = t//3600
    m = (t%3600)//60
    s = (t%3600)%60
    return '%dh:%dm:%ds'%(h,m,s)

def get_curr_datetime():
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H%M")
    return dt_string

def show_prediction(image, gt, model, fig_path, start_time):
    batch_size = len(image)
    # mlb = MultiLabelBinarizer()
    # Generate prediction
    prediction = model.predict(image)
    prediction = np.round(prediction, 5)
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
        # axes[i, 1].axis('off')

        # Display the predictions
        selected = np.where(prediction[i] > 0.5, '*', ' ')
        combined_array = list(zip(LABELS, prediction[i], selected))
        pred_str = '\n'.join([f"{row[2]} {row[0]}, {row[1]}" for row in combined_array])
        axes[i, 2].axis([0, 10, 0, 10])
        axes[i, 2].axis('off')
        axes[i, 2].text(1, 0, pred_str, fontsize=10)
        # axes[i, 2].axis('off')
        
    # style.use('default')
    filename = os.path.join(fig_path, "sample_predict_" + start_time + ".png")
    print("Saving to", filename)
    plt.savefig(filename)
    
    
    
    # plt.figure(figsize=(8,4))
    # plt.imshow(image)
    # plt.title('\n\nGT\n{}\n\nPrediction\n{}\n'.format(gt, list(prediction)), fontsize=9)
    
    # filename = os.path.join(fig_path, "sample_predict.png")
    # print("Saving to", filename)
    # plt.savefig(filename)
    # plt.show()
