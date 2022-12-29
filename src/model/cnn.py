import os
import csv
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import model.utils as utils



# Simple CNN model
class CNN_Model:
    def __init__(self, num_classes, input_shape, learning_rate):
        self._num_classes    = num_classes
        self._input_shape    = input_shape
        self._learning_rate  = learning_rate
        self._test_accuracy  = None
        self._train_accuracy = None
        self._test_loss      = None
        self._train_loss     = None
        self._model          = None

    def built_model(self, use_BN_layers=True):
        self._model = tf.keras.Sequential()
        self._model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                                            input_shape=self._input_shape))
        if use_BN_layers: self._model.add(tf.keras.layers.BatchNormalization())
        self._model.add(tf.keras.layers.Activation('relu'))
        self._model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same'))
        if use_BN_layers: self._model.add(tf.keras.layers.BatchNormalization())
        self._model.add(tf.keras.layers.Activation('relu'))

        self._model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self._model.add(tf.keras.layers.Dropout(0.2))

        self._model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
        if use_BN_layers: self._model.add(tf.keras.layers.BatchNormalization())
        self._model.add(tf.keras.layers.Activation('relu'))
        self._model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
        if use_BN_layers: self._model.add(tf.keras.layers.BatchNormalization())
        self._model.add(tf.keras.layers.Activation('relu'))

        self._model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self._model.add(tf.keras.layers.Dropout(0.3))

        self._model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same'))
        if use_BN_layers: self._model.add(tf.keras.layers.BatchNormalization())
        self._model.add(tf.keras.layers.Activation('relu'))
        self._model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same'))
        if use_BN_layers: self._model.add(tf.keras.layers.BatchNormalization())
        self._model.add(tf.keras.layers.Activation('relu'))

        self._model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self._model.add(tf.keras.layers.Dropout(0.4))

        self._model.add(tf.keras.layers.Flatten())
        self._model.add(tf.keras.layers.Dense(self._num_classes))
        self._model.add(tf.keras.layers.Activation('softmax'))

        self._model.summary()

    def train_model(self, x_train, y_train, x_test, y_test, batch_size, num_epochs,
                    decay=1e-6, data_augmentation=True, show_plot=False):

        opt_rms = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self._model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])

        if data_augmentation:
            # Train model with data augmentation
            data_gen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
            )
            data_gen.fit(x_train)

            model_history=self._model.fit(data_gen.flow(x_train, y_train, batch_size=batch_size),
                                          steps_per_epoch=x_train.shape[0] // batch_size,
                                          epochs=num_epochs, verbose=1,
                                          validation_data=(x_test, y_test))
        else:
            # Train model without data augmentation
            model_history = self._model.fit(x_train, y_train, batch_size=batch_size,
                                            epochs=num_epochs, verbose=1,
                                            validation_data=(x_test, y_test), shuffle=True)

        self._train_loss     = model_history.history['loss'][num_epochs-1]
        self._test_loss      = model_history.history['val_loss'][num_epochs-1]
        self._train_accuracy = model_history.history['accuracy'][num_epochs-1]
        self._test_accuracy  = model_history.history['val_accuracy'][num_epochs-1]

        if show_plot:
            t = time.time()
            # Visualization of loss values during the training process
            path = os.path.join('.', 'model', 'results', 'performance')
            os.makedirs(path, exist_ok=True)
            facts = str(self._learning_rate) + '_' + str(self._train_accuracy) + '_' + str(self._test_accuracy)
            name_1 = str(t) + '_' + 'CNN_Loss_'     + facts +'.png'
            name_2 = str(t) + '_' + 'CNN_Accuracy_' + facts +'.png'

            plt.plot(model_history.epoch, model_history.history['loss'],
                     model_history.epoch, model_history.history['val_loss'])
            plt.title('Loss')
            plt.legend(labels={'training set', 'validation set'})
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.savefig(os.path.join(path, name_1))
            plt.show()

            # Visualization of the accuracy during the training process
            plt.plot(model_history.epoch, model_history.history['accuracy'],
                     model_history.epoch, model_history.history['val_accuracy'])
            plt.title('Accuracy')
            plt.legend(labels={'training set', 'validation set'})
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            plt.savefig(os.path.join(path, name_2))
            plt.show()

    def evaluate_model(self, x_test, y_test):
        scores = self._model.evaluate(x_test, y_test, verbose=1)
        self._test_loss     = scores[0]
        self._test_accuracy = scores[1]
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def get_weights(self):
        weights_by_layer = {}

        for i in range(len(self._model.layers)):
            name = self._model.layers[i].get_config()['name']
            if str(name).find("conv") == 0:
                weights = [self._model.layers[i].get_weights()[0][:, :, :, p] for p in
                           range(self._model.layers[i].get_config()['filters'])]
                bias = self._model.layers[i].get_weights()[1]
                weights_by_layer[name] = {'weights': weights, 'bias': bias}

            if str(name).find("batch") == 0:
                # Batch Normalization Layers have 4 parameters: Gamma and beta are trainable
                # parameters, moving_mean and moving_variance are non trainable parameters
                gamma = K.eval(self._model.layers[i].gamma)
                beta  = K.eval(self._model.layers[i].beta)
                moving_mean = K.eval(self._model.layers[i].moving_mean)
                moving_var  = K.eval(self._model.layers[i].moving_variance)
                weights_by_layer[name] = {'gamma': gamma, 'beta': beta,
                                          'moving_mean': moving_mean, 'moving_var': moving_var}

            if str(name).find("dense") == 0:
                weights = [self._model.layers[i].get_weights()[0][:, u] for u in
                           range(self._model.layers[i].get_config()['units'])]
                bias = self._model.layers[i].get_weights()[1]
                weights_by_layer[name] = {'weights': weights, 'bias': bias}
        return weights_by_layer

    def save_weights(self, save_path, file_name):
        models_weights = self.get_weights()
        layer_list = list(models_weights.keys())
        print("Writing weights to csv file...")

        for l in range(len(self._model.layers)):
            name = self._model.layers[l].get_config()['name']

            if (str(name).find("max_pooling2d") == 0) or (str(name).find("dropout") == 0) or \
               (str(name).find("flatten") == 0) or (str(name).find("activation") == 0):
                continue
            os.makedirs(save_path, exist_ok=True)
            path = os.path.join(save_path, file_name + "_" + name + ".csv")
            csv_file = open(path, "a+")

            with csv_file:
                writer = csv.writer(csv_file, delimiter=';', lineterminator='\n')
                if str(name).find("conv") == 0:
                    # Get weights and biases of current convolutional layer
                    weights_of_all_kernels = self._model.layers[l].get_weights()[0]
                    biases_of_all_kernels  = self._model.layers[l].get_weights()[1]

                    # Create headers
                    k_shape = weights_of_all_kernels.shape
                    headers = utils.create_headers_convLayer(k_shape[:3])
                    writer.writerow(headers)

                    # Write weights to csv-file
                    for kernel in range(k_shape[3]):
                        current_weights   = weights_of_all_kernels[:,:,:,kernel]
                        row               = [kernel, biases_of_all_kernels[kernel]]
                        flattened_weights = current_weights.transpose(2,1,0).transpose(0,2,1).flatten()

                        row.extend(flattened_weights)
                        writer.writerow(row)

                if str(name).find("batch") == 0:
                    # Batch Normalization Layers have 4 parameters: Gamma and beta are trainable
                    # parameters, moving_mean and moving_variance are non trainable parameters
                    gamma       = K.eval(self._model.layers[l].gamma)
                    beta        = K.eval(self._model.layers[l].beta)
                    moving_mean = K.eval(self._model.layers[l].moving_mean)
                    moving_var  = K.eval(self._model.layers[l].moving_variance)

                    # Create headers
                    headers = ['kernel', 'gamma', 'beta', 'moving mean', 'moving variance']
                    writer.writerow(headers)

                    for i in range(len(gamma)):
                        row = [i, gamma[i], beta[i], moving_mean[i], moving_var[i]]
                        writer.writerow(row)

                if str(name).find("dense") == 0:
                    # Get weights and biases of current convolutional layer
                    weights = self._model.layers[l].get_weights()[0]
                    biases  = self._model.layers[l].get_weights()[1]

                    # Create headers
                    w_shape = weights.shape
                    headers = utils.create_headers_denseLayer(w_shape[0])
                    writer.writerow(headers)

                    for i in range(w_shape[1]):
                        row = [str(i), biases[i]]
                        row.extend(weights[:,i])
                        writer.writerow(row)

    def save_model(self, str_file_name):
        model_json = self._model.to_json()
        with open(str_file_name + ".json", "w") as json_file:
            json_file.write(model_json)

        self._model.save_weights(str_file_name + ".h5")
        print("Saved CNN to disk!")

    def load_model(self, str_file_name):
        json_file = open(str_file_name+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self._model = tf.keras.models.model_from_json(loaded_model_json)

        # load weights into new model
        self._model.load_weights(str_file_name+".h5")
        print("Loaded model " + str_file_name + " from disk")

        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def num_classes(self):
        return self._num_classes

    def input_shape(self):
        return self._input_shape

    def model(self):
        return self._model

    def accuracy(self):
        return self._test_accuracy, self._train_accuracy

    def loss(self):
        return self._test_loss, self._train_loss

    def make_prediction(self, x_data, return_categorized=False):
        if len(x_data.shape)<4:
            x = np.expand_dims(x_data, axis=0)
        predictions = self._model.predict(x_data)
        if return_categorized:
            return predictions
        else:
            predicted_classes = []
            for p in predictions:
                class_pred = [np.argmax(p)]
                predicted_classes.append(class_pred)

            return predicted_classes

