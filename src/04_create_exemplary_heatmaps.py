import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from datetime import date

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_relevances', type=str, default='../data/relevance_scores')
    parser.add_argument('--save_path', type=str, default='../data/heatmaps')
    parser.add_argument('--final_layer', type=str, default='conv2d_684')
    # Parameters concerning the decomposition rule of the underlying relevance scores
    parser.add_argument('--rule', type=str, default='epsilon', help='relevance decomposition rule (options: epsilon, alphabeta, gamma)')
    parser.add_argument('--param_val', type=float, default=1e-07)
    parser.add_argument('--target', type=int, default=7)
    parser.add_argument('--sample_name', type=int, default=1000)
    parser.add_argument('--is_adversarial', action='store_true', default=False)
    opt = parser.parse_args()

    # Determine save path
    save_path = save_path = opt.save_path.split('/')
    save_path = os.path.realpath(os.path.join(os.path.dirname(__file__), *save_path))
    save_path = os.path.join(save_path, 'adversarial') if opt.is_adversarial else os.path.join(save_path, 'original')
    file_name = str(date.today()) + '_heatmapLRP_img' + str(opt.sample_name)
    os.makedirs(save_path, exist_ok=True)

    # Read relevance scores
    if opt.is_adversarial:
        root = opt.root_relevances + '/' + 'adversarials' + '/' + opt.rule + '_rule' + '/' + str(opt.param_val)
        df   = utils.read_relevances_of_adversarials(root, [opt.target], opt.final_layer)
        file_name = file_name +'target' + str(opt.target)

    else:
        root = opt.root_relevances + '/' + 'originals' + '/' + opt.rule + '_rule' + '/' + str(opt.param_val)
        df   = utils.read_relevances_of_originals(root, opt.final_layer)

    # Filter relevant data
    if df is not None and len(df) != 0:
        vis_obj = df[df['name'] == opt.sample_name] if opt.sample_name in df['name'].values else None

        if vis_obj is not None:
            # Get values by color channel
            red   = vis_obj.iloc[0, 1:1025].values.tolist()
            green = vis_obj.iloc[0, 1025:2049].values.tolist()
            blue  = vis_obj.iloc[0, 2049:3073].values.tolist()

            # Sum channel values
            l   = [sum(x) for x in zip(red, green, blue)]
            max = max(np.max(l), 1)
            matrix = np.array(l)
            matrix = matrix.reshape((32, 32)) / max

            # Plot
            sns.heatmap(matrix, vmin=-1, vmax=1, annot=False, square=True, cmap='RdBu_r')
            plt.xticks(color='w')
            plt.yticks(color='w')
            plt.tick_params(length=0)
            plt.savefig(os.path.join(save_path, file_name + '.png'), dpi=200)
