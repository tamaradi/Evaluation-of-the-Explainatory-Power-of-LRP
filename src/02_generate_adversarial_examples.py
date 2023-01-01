# CHECK PARAMETER SETTING BEFORE RUNNING!!!
import warnings
import argparse
import os
import tensorflow as tf
import foolbox as fb
import data.data as data
import model.cnn as cnn
import adversarial_attack.utils as aa_utils
from pathlib import Path

# Needed to use TensorflowModel.from_keras(...) to create adversarial examples
tf.compat.v1.disable_eager_execution()
warnings.filterwarnings("ignore")


def main(opt):
    # LOAD CIFAR-10 DATA
    dataset = data.Data(opt.dataset_type, opt.num_classes, opt.path_correctly_classified_imgs, None, opt.load_from_keras)
    images, labels, names = dataset.images_norm(), dataset.labels(), dataset.image_names()
    img_shape  = dataset.image_shape()

    if len(img_shape) != 0:
        # LOAD CNN MODEL
        model_instance = cnn.CNN_Model(opt.num_classes, img_shape, opt.learning_rate)
        model_instance.load_model(opt.model)

        # CREATE FOOLBOX TENSORFLOW MODEL
        model    = model_instance.model()
        fb_model = fb.models.TensorFlowModel.from_keras(model=model, bounds=(0.0, 1.0))
        pred     = model.predict(images)

        # SPLIT DATA
        if opt.is_split:
            split_imgs, split_labels, split_names = aa_utils.split_by_class(images, labels, names, opt.num_classes)
            if opt.all_classes:
                images, labels, names = split_imgs, split_labels, split_names
            else:
                images = [split_imgs[i] for i in opt.classes]
                labels = [split_labels[i] for i in opt.classes]
                names  = [split_names[i] for i in opt.classes]

        # GENERATE ADVERSARIAL EXAMPLES
        aa_utils.create_adversarials(fb_model, images, labels, names, opt.target_classes, opt.dist_type, opt.threshold, opt.max_it,
                            opt.save_path_adv, opt.save_path_dist, opt.save_path_prob, opt.save_path_nones,
                            opt.file_name_dist, opt.file_name_prob, opt.num_classes, opt.is_split)

        # %%
        PATH_1 = "./Results/Adversarials/Test/"
        PATH_2 = "./Results/Pictures_sorted/Correct_Classification/"
        DATA_NAME_MODEL = "./models/finalCNN_250_160_0.001_BNL"
        TARGET = 0
        NUM_CLASSES = 10
        LEARNING_RATE = 0.001

        adversarials_t0 = Data('Adversarial', NUM_CLASSES, PATH_1, TARGET)
        data_test = Data('Test', NUM_CLASSES, PATH_2, None, False)

        # %% LOAD CNN MODEL
        model_instance = CNN_Model(NUM_CLASSES, adversarials_t0.image_shape(), LEARNING_RATE)
        model_instance.load_model(DATA_NAME_MODEL)

        # %% MAKE PREDICTIONS
        prediction = model_instance.make_prediction(adversarials_t0.images(), True)

        # %% ADJUST PROBABILITIES
        NAMES = adversarials_t0.image_names()
        ADVERSARIALS = adversarials_t0.images()
        LABELS = adversarials_t0.labels()
        PATH_3 = "./Results/Probabilities/"

        for n, l, prob in zip(NAMES, LABELS, prediction):
            save_probabilities_of_adv(n, l, TARGET, prob, 'False', 0, 0, 0, PATH_3, '2020-08-21_probabilities_t0')


    else:
        print('No images found!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Parameters concerning the CNN model and the adversarial attack
    parser.add_argument('--model', type=str, default='./model/trained_models/finalCNN_250_160_0.001_BNL')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--target_classes', nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--max_it', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=0.0001)
    parser.add_argument('--dist_type', type=str, default='Mean Abs')

    # Parameter concerning the underlying data and loading process
    parser.add_argument('--load_from_keras', action='store_true', default=False)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--path_correctly_classified_imgs', type=str, default='../data/originals/correctly_classified')
    parser.add_argument('--dataset_type', type=str, default='Test')
    parser.add_argument('--is_split', action='store_true', default=True)
    parser.add_argument('--all_classes', action='store_true', default=False)
    parser.add_argument('--classes', nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Parameters concerning the visualization of the adversarial examples and data saving
    parser.add_argument('--show_orig_img', action='store_true', default=False)
    parser.add_argument('--show_adv_img',  action='store_true', default=False)
    parser.add_argument('--save_adv_img',  action='store_true', default=False)
    parser.add_argument('--save_fig_dist', action='store_true', default=False)
    parser.add_argument('--save_path_adv',  type=str, default='./adversarial_attack/results/adversarials')
    parser.add_argument('--save_path_dist', type=str, default='./adversarial_attack/results/distances')
    parser.add_argument('--save_path_prob', type=str, default='./adversarial_attack/results/probabilities')
    parser.add_argument('--save_path_nones', type=str, default='./adversarial_attack/results/nones')
    parser.add_argument('--file_name_dist',  type=str, default='distances')
    parser.add_argument('--file_name_prob', type=str, default='probabilities')

    opt = parser.parse_args()

    opt.save_path_adv   = opt.save_path_adv + '/' + opt.dataset_type
    opt.save_path_dist  = opt.save_path_dist + '/' + opt.dataset_type
    opt.save_path_prob  = opt.save_path_prob + '/' + opt.dataset_type
    opt.save_path_nones = opt.save_path_nones + '/' + opt.dataset_type

    main(opt)












