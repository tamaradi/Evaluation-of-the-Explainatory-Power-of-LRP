import glob
import os
import pandas as pd

def read_relevances_of_adversarials(root, targets, name_final_layer):
    print("Reading Adversarial Examples' Relevance Scores")
    df_LRP_adversarial = []
    for i, t in enumerate(targets):
        current_paths = glob.glob(root + '/' + 't' + str(t) + '/' + '*' + name_final_layer + '*.csv')
        if not current_paths: continue
        current_path = current_paths[0].replace(os.path.sep, '/')
        path_parts = current_path.split('/')
        current_path = os.path.realpath(os.path.join(os.path.dirname(__file__), *path_parts))
        csv_data = pd.read_csv(current_path, sep=';')
        csv_data.fillna(0)
        csv_data['target'] = t
        csv_data['label']  = -1
        if i == 0:
            df_LRP_adversarial = csv_data
        else:
            df_LRP_adversarial.append(csv_data)
    return df_LRP_adversarial

def read_relevances_of_originals(root, name_final_layer):
    print("Reading Original Images' Relevance Scores")
    df_LRP_original = []
    current_paths = glob.glob(root + '/' + '*' + name_final_layer + '*.csv')
    if current_paths:
        current_path = current_paths[0].replace(os.path.sep, '/')
        path_parts = current_path.split('/')
        current_path = os.path.realpath(os.path.join(os.path.dirname(__file__), *path_parts))
        csv_data = pd.read_csv(current_path, sep=';')
        csv_data.fillna(0)
        csv_data['target'] = -1
        csv_data['label'] = -1
        df_LRP_original = csv_data
    return df_LRP_original

def get_labels_of_orig_images(root, img_names, df_LRP_original, df_LRP_adversarial):
    for index, name in enumerate(img_names):
        path_parts = root.split('/')
        current_path = os.path.realpath(os.path.join(os.path.dirname(__file__), *path_parts))
        img_path = glob.glob(os.path.join(current_path, '*_' + str(name) + '_*.png'))

        if not img_path: continue

        label = img_path[0].rsplit(os.path.sep, 1)
        label = label[1] if len(label) > 1 else 'unkown'
        label = label.split('class_', 1)
        label = label[1] if len(label) > 1 else label[0]
        label = label.split('.')[0]
        label = int(label) if label.isdigit() else -1

        df_LRP_original._set_value(index, 'label', label)

        # Assign true labels to adversarials as well
        for row in df_LRP_adversarial.index[df_LRP_adversarial['name'] == name].tolist():
            df_LRP_adversarial._set_value(row, 'label', label)