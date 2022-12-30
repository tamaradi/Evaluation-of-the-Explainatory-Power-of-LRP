import os
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

EPS = 1e-7

# Function to get model specific information
def get_model_params(model):
  names, weights = [], []
  for layer in model.layers:
    name = layer.name
    names.append(name)
    weights.append(layer.get_weights())
  return names, weights

def predict_labels(model, images, dict_labels):
  preds = model.predict(images)
  labels = decode_predictions(preds, dict_labels)
  return labels

def decode_predictions(preds, dict_labels):
    labels = [dict_labels[int(np.amax(p))] for p in preds]
    return labels

def get_activations(input, model):
  activations = []
  for layer in model.layers:
    keras_function = K.function([model.input], [layer.output])
    activations.append(keras_function([input, 1]))
  return activations

# Function to transfer weights
def rho(weights, gamma, lrp_rule):
  if lrp_rule == 'gamma rule':
    w = tf.convert_to_tensor(weights + gamma*K.maximum(0., weights))
  else:
    w = tf.convert_to_tensor(weights)
  return w

# Function to save relevances in dependency of their underlying layer
def save_relevances(relevances, path, img_name, layer_type):
  csv_file = open(path, "a+")
  with csv_file:
    writer = csv.writer(csv_file, delimiter=';', lineterminator='\n')

    if layer_type=='conv' or layer_type=='pool' or layer_type=='batch':
      kernel_shape = relevances.shape
      if os.stat(path).st_size == 0:
        headers = create_headers_relevances_convLayer(kernel_shape)
        writer.writerow(headers)
      row = [img_name]
      rel = relevances[0].numpy()
      flattened_relevances = rel.transpose(2, 1, 0).transpose(0, 2, 1).flatten()
      row.extend(flattened_relevances)
      writer.writerow(row)

    if layer_type=='dense':
      if os.stat(path).st_size == 0:
        num_weights = relevances.shape[1]
        headers = create_headers_relevances_denseLayer(num_weights)
        writer.writerow(headers)
      rel = relevances[0].numpy()
      row = [img_name]
      row.extend(rel)
      writer.writerow(row)

    if layer_type=='start':
      if os.stat(path).st_size == 0:
        num_weights = relevances.shape[1]
        headers = create_headers_relevances_denseLayer(num_weights)
        writer.writerow(headers)
      row = [img_name]
      row.extend(relevances[0])
      writer.writerow(row)

def create_headers_relevances_convLayer(kernel_shape):
    width, height, depth, num_kernels = kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]
    headers = ['name']
    for k in range(num_kernels):
      for d in range(depth):
        for h in range(height):
          for w in range(width):
            current_header = 'k' + str(k) + '_d' + str(d) + '_w' + str(h) + str(w)
            headers.append(current_header)
    return headers

def create_headers_relevances_denseLayer(num_relevances):
  headers = ['name']
  for w in range(num_relevances):
    current_header = 'r' + str(w)
    headers.append(current_header)
  return headers

def sign(array):
  signs = np.sign(array)
  array_signs = np.where(signs != 0., signs, 1.0)
  return array_signs

