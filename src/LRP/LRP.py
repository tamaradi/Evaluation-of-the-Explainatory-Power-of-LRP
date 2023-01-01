import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops import gen_nn_ops
import LRP.utils as utils

class LayerwiseRelevancePropagation:
    def __init__(self, model, num_classes, gamma=0, alpha=1, epsilon=1e-7, LRP_type='basic', contains_BN=True):
        self.model        = model
        self.num_classes  = num_classes
        self._LRP_type    = LRP_type
        self._contains_BN = contains_BN

        self.epsilon = epsilon
        self.gamma   = gamma
        self.alpha   = alpha
        self.beta    = 1 - alpha

        if LRP_type == 'alpha beta rule':
            self.current_param = 'alpha_' + str(self.alpha)
        elif LRP_type=='gamma rule':
            self.current_param = 'gamma_' + str(self.gamma)
        else:
            self.current_param = 'epsilon_' + str(self.epsilon)

        self.names, self.weights = utils.get_model_params(self.model)
        self.num_layers = len(self.names)

    def compute_relevances(self, image, img_name, path_part_1, target=None, label=None, num_classes=10):
        # Get activations of the image
        activations = utils.get_activations(image, self.model)

        # Save path part 2 & initial relevance score
        starting_layer = self.num_layers - 2
        pre_activation_values = activations[starting_layer][0][0]  # r = self.model.predict(image)
        r = np.zeros(num_classes)
        if target is not None:
            path_part_2 = '_target_' + str(target) + '_' + self._LRP_type  + '.csv'
            # Set initial relevance score
            r[target] = pre_activation_values[target]
            r         = np.expand_dims(r, axis=0)
        elif label is not None:
            path_part_2 = '_originals_' + self._LRP_type + '.csv'
            # Set initial relevance score
            r[label] = pre_activation_values[label]
            r = np.expand_dims(r, axis=0)

        utils.save_relevances(r, path_part_1 + 'start_relevances' + path_part_2, img_name, 'start')

        for i in range(starting_layer, -1, -1):
            name = self.names[i]
            if 'dense' in name:
                r = self.backprop_dense(self.weights[i][0], self.weights[i][1], activations[i - 1][0], r)
                utils.save_relevances(r, path_part_1 + name + path_part_2, img_name, 'dense')
            elif 'flatten' in name:
                r = self.backprop_flatten(activations[i - 1][0], r)
            elif 'pool' in name:
                r = self.backprop_max_pool2d(activations[i - 1][0], r)
                utils.save_relevances(r, path_part_1 + name + path_part_2, img_name, 'pool')
            elif 'conv' in name:
                if i == 0:
                    r = self.backprop_conv2d(self.weights[i][0], self.weights[i][1], image, r)
                else:
                    r = self.backprop_conv2d(self.weights[i][0], self.weights[i][1], activations[i - 1][0], r)
                utils.save_relevances(r, path_part_1 + name + path_part_2, img_name, 'conv')
            elif 'batch' in name:
                r = self.backprop_batchNorm(self.weights[i], activations[i - 1][0], r)
                utils.save_relevances(r, path_part_1 + name + path_part_2, img_name, 'batch')
            else:
                continue

    # Function to propagate relevances backwards in dependency of the layer type
    def backprop_dense(self, w, b, a, r):
        # w: element of the output weights of keras get_weights
        # (which means w=weights[layer_num][0] ---> 0 indicates weights, 1 would
        # indicate bias)
        # r: If dense Layer is the last layer, r is the prediction of the input data
        # b: bias (b=weights[layer_num][1]
        # a: activation of previous layer

        a = tf.convert_to_tensor(a, dtype=tf.float32)

        # Alpha-beta rule
        if self._LRP_type == 'alpha beta rule':
            w_pos, w_neg = K.maximum(w, 0.), K.minimum(w, 0.)
            b_pos, b_neg = K.maximum(b, 0.), K.minimum(b, 0.)
            z_pos = K.dot(a, w_pos) + b_pos + self.epsilon
            s_pos = r / z_pos
            c_pos = K.dot(s_pos, K.transpose(w_pos))
            z_neg = K.dot(a, w_neg) + b_neg - self.epsilon
            s_neg = r / z_neg
            c_neg = K.dot(s_neg, K.transpose(w_neg))
            c     = (self.alpha * c_pos + self.beta * c_neg)
        # Gamma rule
        elif self._LRP_type == 'gamma rule':
            w = tf.convert_to_tensor(w + self.gamma*K.maximum(0., w))
            z = K.dot(a, w) + b
            s = r / (z + self.epsilon*utils.sign(z))
            c = K.dot(s, K.transpose(w))
        # Basic / epsilon rule
        else:
            w = tf.convert_to_tensor(w)
            z = K.dot(a, w) + b
            # Adding epsilon in order to avoid a division by zero
            z_sign = self.epsilon*utils.sign(z)
            s = r / (z + z_sign)
            c   = K.dot(s, K.transpose(w))
        return a * c # Component-wise multiplication

    # Relevance propagation through flatten layer
    def backprop_flatten(self, a, r):
        shape = list(a.shape)
        shape[0] = -1
        return K.reshape(r, shape)[0]

    # Relevance Propagation through max pooling layer
    def backprop_max_pool2d(self, a, r, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1)):
        z = tf.nn.max_pool(a, ksize=ksize[1:-1], strides=strides[1:-1], padding='VALID')

        # Alpha-beta rule
        if self._LRP_type == 'alpha beta rule':
            z_pos, z_neg = K.maximum(z, 0.) + self.epsilon, K.minimum(z, 0.) - self.epsilon
            s_pos = r / z_pos
            c_pos = gen_nn_ops.max_pool_grad_v2(a, z_pos, s_pos, ksize, strides, padding='VALID')
            s_neg = r / z_neg
            c_neg = gen_nn_ops.max_pool_grad(a, z_neg, s_neg, ksize, strides, padding='VALID')
            c     = (self.alpha * c_pos + self.beta * c_neg)
        # Basic / Epsilon rule
        else:
            s = r / (z + self.epsilon*utils.sign(z))
            c = gen_nn_ops.max_pool_grad(a, z, s, ksize, strides, padding='VALID')
        return a * c

    # Relevance propagation through convolutional layer
    def backprop_conv2d(self, w, b, a, r, strides=(1, 1, 1, 1)):

        # Alpha-beta rule
        if self._LRP_type == 'alpha beta rule':
            w_pos, w_neg = K.maximum(w, 0.), K.minimum(w, 0.)
            b_pos, b_neg = K.maximum(b, 0.), K.minimum(b, 0.)
            # Info: conv2d_backprop_input computes the gradients of convolution with respect to the input
            z_pos = K.conv2d(a, kernel=w_pos, strides=strides[1:-1], padding='same') + b_pos + self.epsilon
            s_pos = r / z_pos
            c_pos = tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w_pos, s_pos, strides, padding='SAME')
            z_neg = K.conv2d(a, kernel=w_neg, strides=strides[1:-1], padding='same') + b_neg - self.epsilon
            s_neg = r / z_neg
            c_neg = tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w_neg, s_neg, strides, padding='SAME')
            c     = self.alpha * c_pos + self.beta * c_neg
        # Gamma rule
        elif self._LRP_type == 'gamma rule':
            w = tf.convert_to_tensor(w + self.gamma*K.maximum(0., w))
            z = K.conv2d(a, kernel=w, strides=strides[1:-1], padding='same') + b
            s = r / (z + self.epsilon*utils.sign(z))
            c = K.dot(s, K.transpose(w))
        # Basic / Epsilon rule
        else:
            w = tf.convert_to_tensor(w)
            z = K.conv2d(a, kernel=w, strides=strides[1:-1], padding='same') + b
            s = r / (z + self.epsilon*utils.sign(z))
            c = tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w, s, strides, padding='SAME')
        return a * c

    # Relevance propagation through batch normalization layer
    def backprop_batchNorm(self, w, a, r):
        gamma, beta = w[0], w[1]
        moving_mean, moving_var = w[2], w[3]
        w = gamma / np.sqrt(moving_var + self.epsilon)
        x = a * w
        b = beta - moving_mean * w
        r_x    = (x * r) / (x + b + self.epsilon)
        return r_x

    def predict_labels(self, images, dict_labels):
        return utils.predict_labels(self.model, images, dict_labels)

    def run_lrp(self, images, names, target=None, filepath=None, label=None, num_classes=10):
        print("Running LRP on {0} images...".format(len(images)))
        total_num_imgs = len(images)
        os.makedirs(filepath, exist_ok=True)
        path = filepath + '/' + self.current_param + '_'

        for i in range(total_num_imgs):
            print(str(i+1) + ' of ' + str(total_num_imgs))
            x = np.expand_dims(images[i], axis=0)
            self.compute_relevances(x, names[i], path, target, label, num_classes)