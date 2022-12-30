import os
import numpy as np
import data.utils as utils
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

class Data:
    def __init__(self, dataset_type, number_classes, data_path=None, target=None, load_from_keras=False):
        self._dataset_type = dataset_type
        self._num_classes  = number_classes
        self._target       = target

        if dataset_type=='Train':
            if load_from_keras:
                (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
                self._images, self._labels = x_train, y_train
                self._images_shape = x_train.shape[1:]
                self._no_images    = x_train.shape[0]
                self._names_images = np.array([str(i) for i in range(self._no_images)])
            else:
                self._images, self._labels, self._names_images = utils.load_images_from_folder(data_path + '/' + dataset_type)
                self._images_shape = self._images.shape[1:]
                self._no_images    = self._images.shape[0]

        elif dataset_type=='Test':
            if load_from_keras:
                (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
                self._images, self._labels = x_test, y_test
                self._images_shape = x_test.shape[1:]
                self._no_images    = x_test.shape[0]
                self._names_images = np.array([str(i) for i in range(self._no_images)])
            else:
                self._images, self._labels, self._names_images = utils.load_images_from_folder(data_path + '/' + dataset_type)
                self._images_shape = self._images.shape[1:]
                self._no_images    = self._images.shape[0]

        elif dataset_type=='Adversarial':
            if target is not None:
                self._images, self._names_images = utils.load_adversarials_from_folder(data_path + '/' + 'Target_' + str(target))
                self._images_shape = self._images.shape[1:]
                self._no_images    = self._images.shape[0]
                self._labels = np.array([[target]]*self._no_images)
        else:
            self._images, self._names_images  = None, None
            self._labels       = None
            self._images_shape = None
            self._no_images    = 0

        print(self._no_images, ' Images of Shape ', self._images_shape, '\n')

        self._dictionary = dict({0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                                 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'})

        self._images_norm  = None
        self._labels_cat   = None

        if len(self._images_shape)!= 0:
            self.prep_pixels()
            if target is None:
                self.categorize_labels()

    def prep_pixels(self):
        max_val = np.max(self._images) if self._images.shape[0] != 0  else 1

        if (max_val > 1):
            # Conversion from int to float if needed
            if self._images.dtype != 'float32':
                self._images_norm  = self._images.astype('float32')
            else:
                self._images_norm  = self._images
            # Normalization to range 0-1
            self._images_norm = self._images_norm / 255.0
        else:
            self._images_norm = self._images

    def dictionary_CIFAR10(self):
        return self._dictionary

    def categorize_labels(self):
        if self._labels is not None:
            self._labels_cat = to_categorical(self._labels)

    def num_classes(self):
        return self._num_classes

    def images(self):
        return self._images

    def images_norm(self):
        return self._images_norm

    def labels(self):
        return self._labels

    def labels_cat(self):
        return self._labels_cat

    def image_shape(self):
        return self._images_shape

    def number_images(self):
        return self._no_images

    def image_names(self):
        return self._names_images
