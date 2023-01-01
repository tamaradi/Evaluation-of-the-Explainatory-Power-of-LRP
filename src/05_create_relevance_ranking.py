import argparse
import  os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import utils

def determine_relevance_ranking(df_LRP_adversarial, df_LRP_original, img_names, share_adv, share_orig):
    outside = float('NaN')
    num_elements_adv  = int(round(3072 * max(min(share_adv, 1), 0)))
    num_elements_orig = int(round(3072 * max(min(share_orig, 1), 0)))
    df = pd.DataFrame(columns=['name', 'label', 'target'] + [pos for pos in range(num_elements_orig)])

    for n in img_names:
        sample_orig = df_LRP_original[df_LRP_original['name'] == n]
        sample_adv  = df_LRP_adversarial[df_LRP_adversarial['name'] == n]
        df_adv      = sample_adv.iloc[:, 1:3073]
        df_orig     = sample_orig.iloc[:, 1:3073]

        tops_adv  = pd.DataFrame(df_adv.apply(lambda x: list(df_adv.columns[np.array(x).argsort()[::-1][:num_elements_adv]]), axis=1).to_list())
        tops_orig = pd.DataFrame(df_orig.apply(lambda x: list(df_orig.columns[np.array(x).argsort()[::-1][:num_elements_orig]]), axis=1).to_list())
        old_pos   = [pos for pos in range(num_elements_orig)]

        adv_count = 1
        for i in range(len(tops_adv.index)):
            print(adv_count)
            part_of_tops_adv = tops_adv.iloc[i, :num_elements_adv].to_list()
            new_pos = []

            # Determine if pixel of orig. image is also among the top pixel of adversarial example
            for old in tops_orig.iloc[0, :num_elements_orig].to_list():
                if old in part_of_tops_adv:
                    new_pos.append(part_of_tops_adv.index(old))
                else:
                    new_pos.append(outside)

            diff_list = [n, sample_adv.iloc[i, 3074], sample_adv.iloc[i, 3073]]
            for pos in range(num_elements_orig):
                if new_pos[pos] is outside:
                    diff_list.append(outside)
                else:
                    diff_list.append(new_pos[pos] - old_pos[pos])
            df.loc[len(df)] = diff_list
            adv_count += 1
    return df

def determine_quantile_values(df, share_orig):
    q =  [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    l = []
    n = int(round(3072 * max(min(share_orig, 1), 0)))
    for index in range(n):
        l.extend(df[index].to_list())

    # number of components outside of the scope to be analysed
    if len(l)>0:
        nan_count = np.count_nonzero(np.isnan(l))
        comp_out  = round((nan_count / len(l)) * 100, 4)
        print('Share of relevant original components that do not belong to the most relevant adv. components: ',
              str(comp_out), '%')

        def condition(x):
            return x == 0
        s = sum(condition(x) for x in l)
        comp_zero = round((s / len(l)) * 100, 4)
        print('Share of components that do not undergo any positional changes: ', str(comp_zero), '%')

        print('Quantile values positional shifts:')
        print(np.nanquantile(np.abs(l), q))

def plot_hist(df, share_orig, save_path, file_name, show=False):
    l = []
    n = int(round(3072 * max(min(share_orig, 1), 0)))
    for index in range(n):
        l.extend(df[index].to_list())
    plt.figure(figsize=(6, 6))
    plt.hist(l, bins=61, color="mediumblue", rwidth=0.9)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Relevance Score')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_path, file_name + '.png'), dpi=200)
    if show: plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument('--root_relevances', type=str, default='../data/relevance_scores')
    parser.add_argument('--root_original_imgs', type=str, default='../data/originals/correctly_classified/Test')
    parser.add_argument('--save_path', type=str, default='../data/relevance_ranking')
    parser.add_argument('--final_layer', type=str, default='conv2d_684')
    # Parameters concerning the decomposition rule of the underlying relevance scores
    parser.add_argument('--rule', type=str, default='epsilon', help='relevance decomposition rule (options: epsilon, alphabeta, gamma)')
    parser.add_argument('--param_val', type=float, default=1e-07)
    # Parameters concerning the share of adversarial and original components to be analysed
    parser.add_argument('--share_adv', type=float, default=1)
    parser.add_argument('--share_orig', type=float, default=1)
    # Parameters concerning the underlying adversarial examples
    parser.add_argument('--targets', nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    opt = parser.parse_args()

    root_LRP_adversarials = opt.root_relevances + '/' + 'adversarials' + '/' + opt.rule + '_rule' + '/' + str(opt.param_val)
    root_LRP_originals    = opt.root_relevances + '/' + 'originals' + '/' + opt.rule + '_rule' + '/' + str(opt.param_val)
    save_path = opt.save_path.split('/')
    save_path = os.path.realpath(os.path.join(os.path.dirname(__file__), *save_path))
    file_name = str(date.today()) + '_relevance_ranking_advShare' + str(opt.share_adv) + '_origShare' + str(opt.share_orig)
    os.makedirs(save_path, exist_ok=True)

    # Reading adversarial examples
    df_LRP_adversarial = utils.read_relevances_of_adversarials(root_LRP_adversarials, opt.targets, opt.final_layer)

    # Reading original images
    df_LRP_original = utils.read_relevances_of_originals(root_LRP_originals, opt.final_layer)

    # Determine labels of original images
    img_names = list(df_LRP_original['name'])
    utils.get_labels_of_orig_images(opt.root_original_imgs, img_names, df_LRP_original, df_LRP_adversarial)

    # Determine relevance ranking (positional shift of considered components) & save
    df = determine_relevance_ranking(df_LRP_adversarial, df_LRP_original, img_names, opt.share_adv, opt.share_orig)
    df.to_csv(os.path.join(save_path, file_name + '.csv'), sep=';', index=False)

    # Determine and print quantile values
    determine_quantile_values(df, opt.share_orig)

    # Plot positional shift distribution (hist)
    plot_hist(df, opt.share_orig, opt.save_path, file_name)
