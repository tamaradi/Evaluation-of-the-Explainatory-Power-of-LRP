import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import date
import seaborn as sns
import foolbox as fb
from foolbox.v1.attacks import LBFGSAttack


#---------------------------------- UTILS ADVERSARIAL EXAMPLES ---------------------------------------------------------
# Remove images that belong to the target class of the adversarial attack
def split_by_class(orig_imgs, labels, img_names, num_classes):
    imgs_splited, labels_splited, names_splited = [], [], []
    for c in range(num_classes):
        indices = np.where(labels == [c])[0]
        current_imgs = orig_imgs[indices]
        current_target = labels[indices]
        current_names = img_names[indices]
        imgs_splited.extend([current_imgs])
        labels_splited.extend([current_target])
        names_splited.extend([current_names])
    return imgs_splited, labels_splited, names_splited

def put_splited_data_together(splited_imgs, splited_labels, splited_names, target_class_attack):
    images, labels, names = None, None, None
    for i in range(len(splited_imgs)):
        current_label = splited_labels[i][0][0]
        if current_label is not target_class_attack:
            if images is not None:
                images = np.concatenate((images, splited_imgs[i]), axis=0)
                labels = np.concatenate((labels, splited_labels[i]), axis=0)
                names = np.concatenate((names, splited_names[i]), axis=0)
            else:
                images = splited_imgs[i]
                labels = splited_labels[i]
                names = splited_names[i]
    return images, labels, names

# Function to create instances of foolbox class aversarial based on original images
def create_instances_adversarial_class(fb_model, criterion, orig_imgs, labels, dist_type, threshold):
    # -----------------------------------------------------------------------------------------------------------------------------
    # INPUT DATA:
    # fb_model  : instance of foolbox model
    # criterion : criterion for the misclassification (instance of foolbox class criterion)
    # orig_imgs : original unperturbed images
    # targets   : ground truth target classes of the unperturbed images
    # dist_type : string describing the distance measurement (choices: "Mean Squared", "Mean Abs", "L0", "Linfinity", "ElasticNet")
    # ------------------------------------------------------------------------------------------------------------------------------
    if dist_type == "Mean Squared":
        dist = fb.distances.MeanSquaredDistance
    elif dist_type == "L0":
        dist = fb.distances.L0
    elif dist_type == "Linfinity":
        dist = fb.distances.Linfinity
    elif dist_type == "ElasticNet":
        dist = fb.distances.ElasticNet
    else:
        dist = fb.distances.MeanAbsoluteDistance
    adversarials = []
    total_num_imgs, current_num_imgs = orig_imgs.shape[0], 1
    print("Creating instances of the class Adversarial...")
    for img, label in zip(orig_imgs, labels):
        print(str(current_num_imgs) + " of " + str(total_num_imgs))
        current_adversarial = fb.v1.adversarial.Adversarial(fb_model, criterion, img, label[0],
                                                            distance=dist, threshold=threshold, verbose=False)
        adversarials.append(current_adversarial)
        current_num_imgs += 1
    return adversarials
    # ------------------------------------------------------------------------------------------------------------------------------
    # OUTPUT:
    # adversarials: list of instances of foolbox class Adversarial
    # ------------------------------------------------------------------------------------------------------------------------------


# %% Create adversarials of images for one target class
def create_adversarials_for_one_target(fb_model, orig_imgs, labels, names, target, dist_type,
                                       threshold, max_iter, save_path_img, save_path_dist,
                                       save_path_prob, save_path_nones, file_name_dist, file_name_prob):
    criterion = fb.criteria.TargetClass(target)
    adversarials = create_instances_adversarial_class(fb_model, criterion, orig_imgs, labels, dist_type, threshold)
    num_adv = len(adversarials)
    attack = LBFGSAttack(model=fb_model, criterion=criterion)

    for i in range(num_adv):
        # Determine adversarial example
        print(str(i + 1) + " of " + str(num_adv))
        attack(adversarials[i], unpack=False, maxiter=max_iter)

        if adversarials[i].perturbed is not None:
            # Save distance
            dist = determine_distance_to_origIMG(adversarials[i].unperturbed, adversarials[i].perturbed)
            save_distances_to_orig_img(names[i], labels[i], target, dist, save_path_dist, file_name_dist)

            # Save adversarial als png-file
            save_adversarial(adversarials[i].perturbed, target, save_path_img, names[i])

            # Save propability of adversarial for each class
            reached_threshold = adversarials[i].reached_threshold()
            num_pred_calls = adversarials[i]._total_prediction_calls
            num_gradient_calls = adversarials[i]._total_gradient_calls
            mean_dist = adversarials[i].distance

            save_probabilities_of_adv(names[i], labels[i], target, adversarials[i].output, reached_threshold,
                                      num_pred_calls, num_gradient_calls, mean_dist, save_path_prob, file_name_prob)

        else:
            save_Nones(names[i], labels[i], target, save_path_nones, "Nones_" + str(max_iter) + '_t' + str(target))


# Create adversarials for several targets
def create_adversarials(fb_model, orig_imgs, labels, names, targets, dist_type, threshold,
                        max_iter, save_path_img, save_path_dist, save_path_prob,
                        save_path_nones, file_name_dist, file_name_prob, num_classes, is_splited=False):
    if not is_splited:
        splited_imgs, splited_labels, splited_names = split_by_class(orig_imgs, labels, names, num_classes)
    else:
        splited_imgs, splited_labels, splited_names = orig_imgs, labels, names

    for t in targets:
        print("Determining adversarials with target class " + str(t) + "...")
        current_imgs, current_labels, current_names = put_splited_data_together(splited_imgs, splited_labels,
                                                                                splited_names, t)
        current_file_name_dist = file_name_dist + '_t' + str(t)
        current_file_name_prob = file_name_prob + '_t' + str(t)
        create_adversarials_for_one_target(fb_model, current_imgs, current_labels, current_names, t,
                                           dist_type, threshold, max_iter, save_path_img, save_path_dist,
                                           save_path_prob, save_path_nones, current_file_name_dist,
                                           current_file_name_prob)


# %% Save adversarial as png-file
def save_adversarial(adversarial, target, save_path, img_name):
    # -------------------------------------------------------------------
    # INPUT
    # aversarials:    Instance of class Adversarial of foolbox
    # target_classes: Target class of adversarial attack
    # save_path:      Path, where the adversarial will be saved
    # img_name:       Image name
    # -------------------------------------------------------------------
    path = os.path.join(save_path,'target_' + str(target), 'img_' + img_name + '_t' + str(target) + '.png')
    plt.imsave(fname=path, arr=adversarial, format='png')


# %% Determine the distance between perturbed and unperturbed image
def determine_distance_to_origIMG(original_img, adversarial):
    # ---------------------------------------------------------------------
    # INPUT:
    # original_img: multidimensional array
    # adversarial:  multidimensional array
    # ---------------------------------------------------------------------
    return original_img - adversarial


# %% Save distance of adversarial to its orignal image
def save_distances_to_orig_img(name_img, label, target, distance, save_path, file_name):
    # -----------------------------------------------------------------------
    # INPUT:
    # name_img : name of the image as string
    # label    : ground truth label of the image
    # distance : distance of adversarial to original image (multidim. array)
    # save_path: path of distance file
    # file_name: name of distance file
    # -----------------------------------------------------------------------

    path = os.path.join(save_path, file_name + ".csv")
    csv_file = open(path, "a+")
    with csv_file:
        writer = csv.writer(csv_file, delimiter=';', lineterminator='\n')
        if os.stat(path).st_size == 0:
            num_dists = distance.shape[0] * distance.shape[1]
            headers = create_headers_dist(num_dists)
            writer.writerow(headers)

        row = [name_img, label[0], target]
        flattened_dist = distance.transpose(2, 1, 0).transpose(0, 2, 1).flatten()
        rounded_dist = [str(round(d, 10)) for d in flattened_dist]
        row.extend(rounded_dist)
        writer.writerow(row)


# %% Save probabilities of adversarial for each class
def save_probabilities_of_adv(name_img, label, target, class_prob_adversarial, reached_threshold,
                              num_pred_calls, num_gradient_calls, mean_distance, save_path, file_name):
    # -----------------------------------------------------------------------
    # INPUT:
    # name_img : name of the image as string
    # label    : ground truth label of the image
    # class_prob_adversarial: probability of adversarial for each class
    # save_path: path of probability file
    # file_name: name of probability file
    # -----------------------------------------------------------------------
    path = os.path.join(save_path, file_name + ".csv")
    csv_file = open(path, "a+")

    with csv_file:
        writer = csv.writer(csv_file, delimiter=';', lineterminator='\n')
        if os.stat(path).st_size == 0:
            num_classes = len(class_prob_adversarial)
            headers = create_headers_prop(num_classes)
            writer.writerow(headers)

        row = [name_img, label[0], target, reached_threshold, num_pred_calls,
               num_gradient_calls, mean_distance]
        test = [str(round(i, 10)) for i in class_prob_adversarial]
        row.extend(test)
        writer.writerow(row)


# Save image names for which none adversarial could be found
def save_Nones(name_img, label, target, save_path, file_name):
    # -----------------------------------------------------------------------
    # INPUT:
    # name_img : name of the image as string
    # label    : ground truth label of the image
    # target   : target class of adversarial attack
    # save_path: path of distance file
    # file_name: name of distance file
    # -----------------------------------------------------------------------
    path = os.path.join(save_path, file_name + ".csv")
    csv_file = open(path, "a+")
    with csv_file:
        writer = csv.writer(csv_file, delimiter=';', lineterminator='\n')
        if os.stat(path).st_size == 0:
            headers = ["name", "label", "target"]
            writer.writerow(headers)
        row = [name_img, label[0], target]
        writer.writerow(row)

# Creating an instance of the foolbox LBFGS attack
def LBFGS_attack(target_class, fb_model):
    criterion = fb.criteria.TargetClass(target_class)
    attack = LBFGSAttack(model=fb_model, criterion=criterion)
    return attack


# Create one or more adversarials of a given image
def create_adversarials_of_img(img, true_label, fb_model, max_iter, target_class=None, unpack=False):
    # -------------------------------------------------------------------
    # INPUT
    # img:        Reference image as numpy array
    # true_label: True class of the reference image
    # fb_model:   Instance of a TensorflowModel of foolbox
    # max_iter:   Max. iterations
    # -------------------------------------------------------------------
    # OUTPUT
    # aversarials: List of Instances of class Adversarial of foolbox
    # targets:     List of target classes
    # -------------------------------------------------------------------
    adversarials, targets = [], []
    if target_class is not None:
        attack = LBFGS_attack(target_class, fb_model)
        a = attack(img, label=true_label, unpack=unpack, maxiter=max_iter)
        targets.append(target_class)
        adversarials.append(a)
    else:
        for target in range(0, 10):
            if target != true_label:
                attack = LBFGS_attack(target, fb_model)
                a = attack(img, label=true_label, unpack=unpack, maxiter=max_iter)
                targets.append(target)
                adversarials.append(a)
    return adversarials, targets


# Create one or more adversarials of the given images
def create_adversarials_for_data(images, true_labels, fb_model, max_iter, save_path_ad=None, save_adversarials=False,
                                 save_path_dist=None, file_name_dist=None, num_dists=0, save_ad_dist=False,
                                 dataset_type=None, img_names=None, target_class=None, unpack=False):
    n_imgs, it_current_img = len(images), 1
    print("Create adversarial examples of " + str(n_imgs) + " images...")
    true_labels_array = [true_labels[i][0] for i in range(len(true_labels))]
    adversarials, targets = [], []

    if target_class is not None:
        attack = LBFGS_attack(target_class, fb_model)
        for img, label, name in zip(images, true_labels_array, img_names):
            print("Image " + str(it_current_img) + " of " + str(n_imgs) + " images...")
            if label != target_class:
                a = attack(img, label=label, unpack=unpack, maxiter=max_iter)

                if not save_adversarials and not save_ad_dist:
                    adversarials.append([a]), targets.append([target_class])

                if save_adversarials:
                    save_all_adversarials_of_one_img([a], [target_class], save_path_ad, dataset_type, name)

                if save_ad_dist:
                    # Determine distances to original image
                    dist = determine_distances_to_origIMG(img, [a])
                    save_distances_to_orig_imgs([name], [label], [target_class], dist, num_dists,
                                                save_path_dist, dataset_type, file_name_dist)
            it_current_img += 1

    else:
        for img, label, name in zip(images, true_labels_array, img_names):
            print("Image " + str(it_current_img) + " of " + str(n_imgs) + " images...")
            a, t = create_adversarials_of_img(img, label, fb_model, max_iter, target_class=target_class, unpack=unpack)

            if not save_adversarials and not save_ad_dist:
                adversarials.append(a), targets.append(t)

            if save_adversarials:
                save_all_adversarials_of_one_img(a, t, save_path_ad, dataset_type, name)

            if save_ad_dist:
                # Determine distances to original image
                dist = determine_distances_to_origIMG(img, a)
                save_distances_to_orig_imgs([name], [label], [t], [dist], num_dists,
                                            save_path_dist, dataset_type, file_name_dist)

            it_current_img += 1
    return adversarials, targets


def save_all_adversarials_of_one_img(adversarials, target_classes, save_path, dataset_type, img_name):
    # -------------------------------------------------------------------
    # INPUT
    # aversarials:    List of Instances of class Adversarial of foolbox
    # target_classes: List of target classes
    # save_path:      Path, where the adversarials will be saved
    # dataset_type:   Possible types are 'Train' and 'Test'
    # img_name:       Image name
    # -------------------------------------------------------------------
    adv_array = [adversarials[i].perturbed for i in range(len(adversarials))]

    print('Saving all adversarials of image ' + img_name)
    for ad_ex, target in zip(adv_array, target_classes):
        if ad_ex is not None:
            path = save_path + dataset_type + '/Target_' + str(target) + '/img_' + img_name + '_t' + str(
                target) + '.png'
            plt.imsave(fname=path, arr=ad_ex, format='png')

# Functions to determine distances to original input image
def determine_distance_to_origIMG(original_img, adversarial):
    if adversarial is not None:
        return original_img - adversarial

def determine_distances_to_origIMG(original_img, adversarials):
    distances = []
    for a in adversarials:
        dist = determine_distance_to_origIMG(original_img, a.perturbed)
        distances.append(dist)
    return distances


def determine_distances_to_allIMGs(original_imgs, adversarials):
    all_distances = []
    for img, a in zip(original_imgs, adversarials):
        current_distances = determine_distances_to_origIMG(img, a)
        all_distances.append(current_distances)
    return all_distances


# Save distances to original images as csv file
def save_distances_to_orig_imgs(names_orig_imgs, true_classes, target_classes, distances, num_dists,
                                save_path, dataset_type, file_name):
    print("Writing distances to csv file...")
    path = os.path.join(save_path, dataset_type, file_name + ".csv")
    csv_file = open(path, "a+")

    with csv_file:
        writer = csv.writer(csv_file, delimiter=';', lineterminator='\n')
        if os.stat(path).st_size == 0:
            headers = create_headers_dist(num_dists)
            writer.writerow(headers)

        for n, l, targets, dists in zip(names_orig_imgs, true_classes, target_classes, distances):
            for t, d in zip(targets, dists):
                row = [n, l, t]

                if d is not None:
                    for d_r in d[:, :, 0].flatten():
                        row.append(d_r)
                    for d_g in d[:, :, 1].flatten():
                        row.append(d_g)
                    for d_b in d[:, :, 2].flatten():
                        row.append(d_b)
                else:
                    for i in range(0, num_dists):
                        row.append(None)
                writer.writerow(row)


# Funktions to visualize adversarial examples
def visualize_adversarials_allClasses(original_img, true_label, adversarials, target_classes,
                                      img_name, save_path=None, save_fig=False):
    nrows, ncols, figsize = 2, 5, [12, 6]
    adv_array = [adversarials[i].perturbed for i in range(len(adversarials))]
    outputs = [np.max(adversarials[i].output) for i in range(len(adversarials))]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i:   Runs from 0 to (nrows*ncols-1)
        # axi: Equivalent with ax[rowid][colid]
        if i != 0:
            axi.imshow(adv_array[i - 1], interpolation='nearest')
            axi.set_title("Target class: " + str(target_classes[i - 1]) + " (" + str(round(outputs[i - 1], 4)) + ")",
                          fontsize=10)
        else:
            axi.imshow(original_img, interpolation='nearest')
            axi.set_title("Original class: " + str(true_label), fontsize=10)

    fig.suptitle("Adversarials of image " + img_name, fontsize=12)
    plt.tight_layout(True)

    if save_fig:
        d = date.today()
        path = save_path + str(d) + "_alltargets_" + img_name + ".png"
        plt.savefig(path, dpi=100)
    plt.show()


def visualize_adversarial(original_img, true_label, adversarial, target_class, img_name,
                          save_path=None, save_fig=False, show_original_img=False):
    title = "Adversarial example of image " + img_name
    if not show_original_img:
        fig = plt.figure(figsize=(6, 5))
        gs = gridspec.GridSpec(1, 1, width_ratios=[1])
        ax = plt.subplot(gs[0])
    else:
        fig = plt.figure(figsize=(8, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 2])
        ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])

        ax1.imshow(original_img, interpolation='nearest')
        ax1.set_title("Original class: " + str(true_label), fontsize=10)
        ax2.imshow(adversarial.perturbed, interpolation='nearest')
        ax2.set_title("Target class: " + str(target_class), fontsize=10)

    fig.suptitle(title, fontsize=12)
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])

    if save_fig and save_path is not None:
        d = date.today()
        path = save_path + str(d) + "_adex_t" + str(target_class) + "_img_" + img_name + ".png"
        plt.savefig(path, dpi=100)
    plt.show()


# %% Functions to visualize distances
def visualize_distance(distance, target_class, name_orig_img, save_fig=False, save_path=None,
                       red_type='absolut', option='sum', cmap_type='RdBu_r', ):
    title = "Difference of adversarial and image " + name_orig_img + "(target class " + str(target_class) + ")"
    reduced_dist = reduce_channels_distance(distance, axis=-1, red_type=red_type, option=option)

    fig = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(1, 1, width_ratios=[1])
    ax = plt.subplot(gs[0])

    min, max = np.amin(reduced_dist), np.amax(reduced_dist)

    fig.suptitle(title, fontsize=12)
    sns.heatmap(reduced_dist, vmin=min, vmax=max, cmap=cmap_type, ax=ax)
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])

    if save_fig and save_path is not None:
        d = date.today()
        path = save_path + str(d) + "_dist_adex_" + target_class + "_img_" + name_orig_img + ".png"
        plt.savefig(path, dpi=100)
    plt.show()

def visualize_distances(original_img, true_label, distances, target_classes, img_name, save_fig=False,
                        save_path=None, cmap_type='RdBu_r', red_type='absolut', option='sum'):
    title = "Difference of adversarials and image " + img_name
    nrows, ncols, figsize = 2, 5, [12, 6]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i:   Runs from 0 to (nrows*ncols-1)
        # axi: Equivalent with ax[rowid][colid]
        if i != 0:
            red_dist = reduce_channels_distance(distances[i - 1], red_type=red_type, option=option)
            # min, max = np.amin(red_dist), np.amax(red_dist)
            min, max = 0, 1
            sns.heatmap(red_dist, vmin=min, vmax=max, cmap=cmap_type, ax=axi)
            axi.set_title("Target class: " + str(target_classes[i - 1]) + ")", fontsize=10)
        else:
            axi.imshow(original_img, interpolation='nearest')
            axi.set_title("Original class: " + str(true_label), fontsize=10)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(True)

    if save_fig and save_path is not None:
        d = date.today()
        path = save_path + str(d) + "_distances_img_" + img_name + ".png"
        plt.savefig(path, dpi=100)
    plt.show()

def visualize_img_adversarial_distances(img, adversarial, distances, img_name, true_label, target_class,
                                        save_fig=False, save_path=None, cmap_type='RdBu_r', red_type='absolut',
                                        option='sum'):
    title = 'Image ' + str(img_name) + ', True class: ' + str(true_label) + ', Target class: ' + str(target_class)
    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 3])
    ax1, ax2, ax3 = plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])

    red_dist = reduce_channels_distance(distances, red_type=red_type, option=option)
    min, max = np.amin(red_dist), np.amax(red_dist)

    fig.suptitle(title, fontsize=12)
    ax1.imshow(img, interpolation='nearest')
    ax1.set_title('Original image', fontsize=10)
    sns.heatmap(red_dist, vmin=min, vmax=max, cmap=cmap_type, ax=ax2)
    ax2.set_title('Differences', fontsize=10)
    sns.imshow(adversarial, interpolation='nearest')
    ax3.set_title('Adversarial example', fontsize=10)

    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])

    if save_fig and save_path is not None:
        d = date.today()
        path = save_path + str(d) + "_ad_t" + str(target_class) + "_dist_img_" + img_name + ".png"
        plt.savefig(path, dpi=100)
    plt.show()

# Reduce channels of distance tensor to display as heatmap
def reduce_channels_distance(distances, axis=-1, red_type='absolut', option='sum'):
    # red_type: Typ of reduction ('relativ' or 'absolut')
    # option:   Possible choices --> 'sum'or 'mean'
    if red_type == 'absolut':
        reduced_dist = abs(distances)
    else:
        reduced_dist = distances

    if option == 'sum':
        return reduced_dist.sum(axis=axis)
    elif option == 'mean':
        return reduced_dist.mean(axis=axis)
    else:
        return distances

def create_headers_dist(num_dists_per_channel):
    headers = ["name", "label", "target"]
    for c in ['r_', 'g_', 'b_']:
        for i in range(num_dists_per_channel):
            h = c + str(i)
            headers.append(h)
    return headers

def create_headers_prop(num_classes):
    headers = ["name", "label", "target", "reached threshold", "pred calls", "gradient calls", "mean dist"]
    for i in range(num_classes):
        h = "c_" + str(i)
        headers.append(h)
    return headers