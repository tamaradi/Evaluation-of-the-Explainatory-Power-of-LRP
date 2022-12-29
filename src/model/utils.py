import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import date


#-------------------------------------------------- UTILS CNN ----------------------------------------------------------
# Determines which images of X_data are correctly classified by the given model and which
# aren't and returns an array of boolean values
def determine_correctness_of_classification(CNN_Instance, X_data, Y_data, return_categorized=False):
    print("Determine correctness of image classification...")
    pred_data = CNN_Instance.make_prediction(X_data, return_categorized)
    class_is_correct = np.equal(pred_data, Y_data)

    return class_is_correct

# Divide images into correctly and incorrectly classified images and save them
def divide_and_save_by_correct_classification(X_data, Y_data, correctness_list, path_correct, path_incorrect,
                                              folder, name_data_set, indices_only=False):
    # Define parts of file paths
    subpath_correct   = os.path.join(path_correct, folder)
    subpath_incorrect = os.path.join(path_incorrect, folder)

    os.makedirs(subpath_correct, exist_ok=True)
    os.makedirs(subpath_incorrect, exist_ok=True)

    # Save images in different folders by correctness of their classification if stated
    if not indices_only:
        print("Dividing and saving images by correctness of their classification...")
        for index in range(len(correctness_list)):
            if correctness_list[index]:
                path = os.path.join(subpath_correct, "img_" + str(index) + "_class_" + str(Y_data[index][0]) + ".png")
            else:
                path = os.path.join(subpath_incorrect, "img_" + str(index) + "_class_" + str(Y_data[index][0]) + ".png")
            plt.imsave(fname=path, arr=X_data[index], format='png')

    # Filter indices of correct and incorrect classified images
    print("Save indices of correct and incorrect classified images...")
    indices_correct   = np.where(correctness_list == [True])[0]
    indices_incorrect = np.where(correctness_list == [False])[0]
    d = date.today()
    path_corr   = os.path.join(path_correct, str(d) + "_indices_correct"   + name_data_set + ".csv")
    path_incorr = os.path.join(path_incorrect, str(d) + "_indices_incorrect" + name_data_set + ".csv")
    with open(path_corr, "w") as file_c:
        for i in indices_correct:
            file_c.write('%s\n' % i)
    with open(path_incorr, "w") as file_inc:
        for i in indices_incorrect:
            file_inc.write('%s\n' % i)

# Creating headers to save weights
def create_headers_convLayer(kernel_shape):
    width, height, depth = kernel_shape[0], kernel_shape[1], kernel_shape[2]
    headers = ['kernel', 'bias']
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                current_header = 'd' + str(d) + '_w' + str(h) + str(w)
                headers.append(current_header)
    return headers

def create_headers_denseLayer(num_nodes):
    headers = ['class', 'bias']
    for i in range(num_nodes):
        headers.append('w_' + str(i))
    return headers
