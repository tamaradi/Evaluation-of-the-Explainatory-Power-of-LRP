# CHECK PARAMETER SETTING BEFORE RUNNING!!!
import time
import os
import argparse
import tensorflow as tf
import numpy as np
import model.cnn as cnn
import model.utils as utils
import data.data as data

def main(opt):
    # Loading CIFAR data
    train = data.Data('Train', opt.num_classes, None, None, opt.load_from_keras)
    test  = data.Data('Test', opt.num_classes, None, None, opt.load_from_keras)


    X_train, Y_train          = train.images(), train.labels()
    X_test,  Y_test           = test.images(), test.labels()
    X_train_norm, Y_train_cat = train.images_norm(), train.labels_cat()
    X_test_norm,  Y_test_cat  = test.images_norm(), test.labels_cat()
    input_shape  = train.image_shape()

    # Build model and train
    if opt.train_model:
        for epoch in opt.num_epochs:
            for bsize in opt.batch_sizes:
                for rate in opt.learning_rates:
                    # Setting random seed
                    np.random.seed(opt.random_seed), tf.compat.v1.random.set_random_seed(opt.random_seed + 1)

                    # Build CNN Model with chosen parameters
                    model = cnn.CNN_Model(opt.num_classes, input_shape, rate)
                    model.built_model()

                    # Train CNN Model
                    model.train_model(X_train_norm, Y_train_cat, X_test_norm, Y_test_cat,
                                      bsize, epoch, 1e-6, opt.augmentation, opt.show_plot)

                    # Evaluate CNN Model
                    model.evaluate_model(X_test_norm, Y_test_cat)

                    # Save CNN Model
                    t = time.time()
                    facts = str(epoch) + '_' + str(bsize) + '_' + str(rate)
                    path = os.path.join('.', 'model', 'trained_models')
                    os.makedirs(path, exist_ok=True)
                    if opt.contains_BNLayer:
                        model.save_model(os.path.join(path, str(t) + '_' + facts + "_simple_CNN_BNL"))
                    else:
                        model.save_model(os.path.join(path, str(t) + '_' + facts + "_simple_CNN_BNL"))

    # Load model
    if not opt.train_model:
        model = cnn.CNN_Model(opt.num_classes, input_shape, opt.learning_rates)
        model.load_model(opt.model)

    # Predict class of train and test data and divide them (e.g., check which images are correctly classified and
    # which aren't)
    if opt.check_classification:
        class_is_correct_train = utils.determine_correctness_of_classification(model, X_train_norm, Y_train, opt.return_categorized)
        class_is_correct_test  = utils.determine_correctness_of_classification(model, X_test_norm, Y_test, opt.return_categorized)

    # Divide images in correctly classified and incorrectly classified images and save them
    if opt.divide_data and opt.check_classification:
        utils.divide_and_save_by_correct_classification(X_train_norm, Y_train, class_is_correct_train,
                                                  opt.path_correctly_classified_imgs, opt.path_incorrectly_classified_imgs,
                                                  'Train', 'train', opt.indices_only)
        utils.divide_and_save_by_correct_classification(X_test_norm, Y_test, class_is_correct_test,
                                                  opt.path_correctly_classified_imgs, opt.path_incorrectly_classified_imgs,
                                                  'Test', 'test', opt.indices_only)
    # Save weights
    if opt.save_weights:
        model.save_weights(os.path.join(opt.path_weights, str(time.time())), opt.file_name_weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Parameters concerning the CNN architecture as well as the training process
    parser.add_argument('--model', type=str, default='./model/trained_models/finalCNN_250_160_0.001_BNL')
    parser.add_argument('--num_epochs', nargs='+', default=[1])
    parser.add_argument('--batch_sizes', nargs='+', default=[64])
    parser.add_argument('--learning_rates', type=float, default=[0.001])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--show_plot', action='store_true', default=True)
    parser.add_argument('--load_from_keras', action='store_true', default=True)
    parser.add_argument('--train_model', action='store_true', default=True)
    parser.add_argument('--contains_BNLayer', action='store_true', default=True)

    # Parameters concerning the sorting out of correctly classified images
    parser.add_argument('--check_classification', action='store_true', default=False)
    parser.add_argument('--divide_data', action='store_true', default=False)
    parser.add_argument('--return_categorized', action='store_true', default=False)
    parser.add_argument('--indices_only', action='store_true', default=False) # Whether to save correctly and
    # incorrectly classified images and their indices or indices only
    parser.add_argument('--path_correctly_classified_imgs', type=str, default='./model/results/data_sorted/correct_classification')
    parser.add_argument('--path_incorrectly_classified_imgs', type=str, default='./model/results/data_sorted/incorrect_Classification')

    # Parameters concerning the saving of weights
    parser.add_argument('--save_weights', action='store_true', default=True)
    parser.add_argument('--path_weights', type=str, default='./model/results/weights')
    parser.add_argument('--file_name_weights', type=str, default='weights')

    opt = parser.parse_args()
    main(opt)

