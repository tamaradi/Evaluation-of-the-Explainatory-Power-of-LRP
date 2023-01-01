# CHECK PARAMETER SETTING BEFORE RUNNING!!!
import argparse
import os
import model.cnn as cnn
import data.data as data
import LRP.LRP as LRP


def main(opt):
    datasets, images, labels, names = [], [], [], []

    # Load data
    if opt.dataset_type == 'Adversarials':
        for t in opt.targets:
            dataset = data.Data(opt.dataset_type, opt.num_classes, opt.target_path, t, opt.load_from_keras)
            datasets.append(dataset)
            images.append(dataset.images_norm())
            labels.append(dataset.labels())
            names.append(dataset.image_names())
    else:
        dataset = data.Data(opt.dataset_type, opt.num_classes, opt.target_path, None, opt.load_from_keras)
        datasets.append(dataset)
        images.append(dataset.images_norm())
        labels.append(dataset.labels())
        names.append(dataset.image_names())

    if len(images) != 0:
        img_shape = datasets[0].image_shape()

        # Load CNN model
        model = cnn.CNN_Model(opt.num_classes, dataset.image_shape(), opt.learning_rate)
        model.load_model(opt.model)

        # LAYER-WISE RELEVANCE PROPAGATION
        if opt.LRP_type == 'basic':
            # Run LRP with epsilon for all targets:
            for e in opt.epsilons:
                LRP_result = LRP.LayerwiseRelevancePropagation(model.model(), model.num_classes(), 0, 1, e, opt.LRP_type,
                                                        opt.contains_BN_layer)
                if opt.targets:
                    for index in range(len(opt.targets)):
                        print('LRP for target ' + str(opt.targets[index]) + '...')
                        p = opt.save_path + '/' + 'epsilon_rule' + '/' + str(e) + '/' + 't' + str(index)
                        LRP_result.run_lrp(images[index], names[index], opt.targets[index], p)
                else:
                    print('LRP for original images...')
                    p = opt.save_path + '/' + 'epsilon_rule' + '/' + str(e)
                    LRP_result.run_lrp(images[0], names[0], None, p, labels[0])

        if opt.LRP_type == 'alpha beta rule':
            # Run LRP with alpha beta for all targets:
            for a in opt.alphas:
                LRP_result = LRP.LayerwiseRelevancePropagation(model.model(), model.num_classes(), 0, a,
                                                    1e-7, opt.LRP_type, opt.contains_BN_layer)
                if opt.targets:
                    for index in range(len(opt.targets)):
                        print('LRP for target ' + str(opt.targets[index]) + ' and alpha ' + str(a) + '...')
                        p = opt.save_path + '/' + 'alphabeta_rule' + '/' + str(a) + '/' + 't' + str(index)
                        LRP_result.run_lrp(images[index], names[index], opt.targets[index], p)
                else:
                    print('LRP for original images...')
                    p = opt.save_path + '/' + 'alphabeta_rule' + '/' + str(a)
                    LRP_result.run_lrp(images[0], names[0], None, p, labels[0])

        if opt.LRP_type == 'gamma rule':
            # Run LRP with gamma rule for all targets:
            for g in opt.gammas:
                LRP_result = LRP.LayerwiseRelevancePropagation(model.model(), model.num_classes(), g, 1,
                                                                1e-7, opt.LRP_type, opt.contains_BN_layer)
                if opt.targets:
                    for index in range(len(opt.targets)):
                        print('LRP for target ' + str(opt.targets[index]) + ' and gamma ' + str(g) + '...')
                        p = opt.save_path + '/' + 'gamma_rule' + '/' + str(g) + '/' + 't' + str(index)
                        LRP_result.run_lrp(images[index], names[index], opt.targets[index], p)
                else:
                    p = opt.save_path + '/' + 'gamma_rule' + '/' + str(g)
                    LRP_result.run_lrp(images[0], names[0], None, p, labels[0])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_from_keras', action='store_true', default=False)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--targets', nargs='+', default=[])
    parser.add_argument('--dataset_type', type=str, default='Test')
    parser.add_argument('--target_path', type=str, default='../data/originals/correctly_classified')
    # Parameters concerning the CNN architecture
    parser.add_argument('--model', type=str, default='./model/trained_models/finalCNN_250_160_0.001_BNL')
    parser.add_argument('--contains_BN_layer', action='store_true', default=True)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # Parameters concerning Layerwise Relevance Propagation
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--gammas', nargs='+', default=[0])
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--alphas', nargs='+', default=[1])
    parser.add_argument('--epsilon', type=float, default=1e-7)
    parser.add_argument('--epsilons', nargs='+', default=[1e-7])
    parser.add_argument('--LRP_type', type=str, default='basic')  # Possibilities: 'basic', 'gamma rule', 'alpha beta rule'
    parser.add_argument('--output_range', type=tuple, default=(-1, 1))
    parser.add_argument('--save_path', type=str, default='../data/relevance_scores/originals')

    opt = parser.parse_args()
    os.makedirs(opt.save_path, exist_ok=True)
    main(opt)
    print('Done!')



