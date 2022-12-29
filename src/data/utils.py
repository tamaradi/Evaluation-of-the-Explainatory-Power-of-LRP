import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#------------------------------------------------- UTILS DATA ----------------------------------------------------------
# Function Image Visualization
def show_data_images(x_data, y_data, dictionary=dict(), columns=3):
    if columns <= 0:
        columns = 1
    if len(x_data.shape) == 3:
        if len(dictionary) == 0:
            label = y_data[0]
        else:
            label = dictionary[y_data[0]]
        plt.title(label)
        plt.imshow(x_data)
    elif x_data.shape[0] == 1:
        if len(dictionary) == 0:
            label = y_data[0][0]
        else:
            label = dictionary[y_data[0][0]]
        plt.title(label)
        plt.imshow(x_data[0])
    else:
        num_images = x_data.shape[0]
        quotient_im_col = float(num_images) / columns

        if quotient_im_col % 1 == 0:
            rows = int(quotient_im_col)
        else:
            rows = int(quotient_im_col) + 1

        index_image = 0
        fig, ax_array = plt.subplots(rows, columns, squeeze=False)

        for i, ax_row in enumerate(ax_array):
            for j, axes in enumerate(ax_row):
                if len(dictionary) == 0:
                    label = y_data[index_image][0]
                else:
                    label = dictionary[y_data[index_image][0]]

                axes.set_title(label)
                axes.imshow(x_data[index_image])
                index_image = index_image + 1

                if index_image == num_images:
                    break
    plt.show()

def load_indices(path):
    indices = []
    with open(path, 'r') as csv_file:
        for line in csv_file:
            index = int(line.rstrip('\n'))
            indices.append(index)
    return indices

# Load images from folder
def load_images_from_folder(path_folder_imgs):
    print("Reading images from  " + path_folder_imgs)
    images, labels, img_names = [], [], []
    for filename in os.listdir(path_folder_imgs):
        filename_split = filename.split('_')
        img_name = filename_split[1]
        label    = int(filename_split[-1].split('.')[0])
        img = mpimg.imread(os.path.join(path_folder_imgs, filename))
        if img is not None:
            images.append(img[:, :, 0:3])
            labels.append([label])
            img_names.append(img_name)
    img_array, labels_array, img_names_array = np.array(images), np.array(labels), np.array(img_names)
    return img_array, labels_array, img_names_array

def load_adversarials_from_folder(path_folder_adv):
    print("Reading adversarials from  " + path_folder_adv)
    adversarials, adv_names = [], []
    for filename in os.listdir(path_folder_adv):
        filename_split = filename.split('_')
        adv_name       = filename_split[1]
        adv = mpimg.imread(os.path.join(path_folder_adv, filename))
        if adv is not None:
            adversarials.append(adv[:, :, 0:3])
            adv_names.append(adv_name)
    adv_array, adv_name_array = np.array(adversarials), np.array(adv_names)
    return adv_array, adv_name_array

#---------------------------------------------- UTILS READ & WRITE -----------------------------------------------------

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
