import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
import PIL
from PIL import Image


def results_visualization(root_dir):
    '''
        Visualization of all run's results, which are recorded in the results table,
        in all subdir in root_dir.
    '''

    dirlist1 = os.listdir(root_dir)
    dirlist_tot = []
    for dirname in dirlist1:
        if len(dirname.split('.')) < 2:
            dirlist_tot.append(dirname)

    if len(dirlist_tot) > 0:
        num_cols = 0
        num_rows_max = 5
        dirlist = []
        for dirr in dirlist_tot:
            results_table_path = os.path.join(os.path.join(root_dir, dirr), 'results.csv')
            if os.path.isfile(results_table_path):
                results_table = pd.read_csv(results_table_path, index_col=0)
                if len(results_table.index) > 0:
                    dirlist.append(dirr)
                    num_cols = len(results_table.columns) - 1

        if len(dirlist) > 0 and num_cols > 0:
            num_rows = num_rows_max
            num_loops = int(np.ceil(len(dirlist) / float(num_rows_max)))

            for loop_num in range(num_loops):
                print('starting making results visualization {}'.format(loop_num+1))
                idx_min = loop_num*num_rows_max
                idx_max = min(loop_num*num_rows_max + num_rows_max, len(dirlist))

                f, axarr = plt.subplots(num_rows, num_cols)
                f.set_size_inches((19, 23), forward=False)
                plt.suptitle('Results visualization', fontsize=16)
                for i, exp in enumerate(dirlist[idx_min:idx_max]):
                    results_table_path = os.path.join(os.path.join(root_dir, exp), 'results.csv')
                    if os.path.isfile(results_table_path):
                        results_table = pd.read_csv(results_table_path, index_col=0)
                        if len(results_table.index) > 0:
                            col_list = list(results_table)
                            assert 'iter' in col_list
                            x = np.array(results_table['iter'])
                            j = 0
                            for col in col_list:
                                if j == 0:
                                    axarr[i, 0].set_title('Experiment {}'.format(exp), fontsize=14)
                                if col != 'iter':
                                    y = np.array(results_table[col])
                                    if col == 'train_loss':
                                        y_best = np.min(y)
                                    else:
                                        y_best = np.max(y)
                                    axarr[i, j].plot(x, y)
                                    axarr[i, j].set_ylabel(col)
                                    axarr[i, j].text(1, 0.5, 'best {}: {:.4f}'.format(col, y_best),
                                                     verticalalignment='bottom', horizontalalignment='right',
                                                     transform=axarr[i, j].transAxes, color='green', fontsize=12)
                                    j += 1

                # Fine-tune figure; make subplots farther from each other.
                f.subplots_adjust(hspace=0.3)

                # save the figure to file
                path = os.path.join(root_dir, 'results_{}.png'.format(loop_num+1))
                f.savefig(path, dpi=500, bbox_inches='tight')
                print('visualization is saved to {}'.format(root_dir))
                # plt.show()
        else:
            print('no valid results table found in the dir {}'.format(root_dir))


def w_params_visualization(train_dir, num_workers):
    '''
        Visualization of advancement of all workers in PBT training
    '''
    history_path = os.path.join(train_dir, 'history.csv')
    if os.path.isfile(history_path):
        history_table = pd.read_csv(history_path, index_col=0)
        num_rows, num_cols = history_table.shape
        assert num_rows > 0
        num_rows_per_worker = int(num_rows / float(num_workers))

        # get param names
        # num_params = (num_cols - 4) // 2
        param_list = []
        for col in list(history_table):
            if col != 'worker' and col != 'epoch' and col != 'iter' and col != 'test_acc' and col != 'copied_from_w' \
                    and not col.startswith('mutation_'):
                param_list.append(col)
        num_params = len(param_list)
        print('num_cols = {}, num_params = {}'.format(num_cols, num_params), 'param_list = ', param_list)

        # every worker array will have two params (x, y and accuracy), if there are more params, they will be added
        ## to x or y for 2D representation purposes
        num_params_repr = 2
        if num_params == num_params_repr:
            worker_arrays = np.zeros((num_workers, num_rows_per_worker, num_params_repr+1))

            w_row_idx = -1
            for row_idx in range(num_rows):
                w = int(history_table.iloc[row_idx]['worker'])
                if w == 0:
                    w_row_idx += 1
                for param_idx in range(num_params):
                    worker_arrays[w][w_row_idx][param_idx%num_params_repr] += history_table.iloc[row_idx][param_list[param_idx]]
                    worker_arrays[w][w_row_idx][2] = history_table.iloc[row_idx]['test_acc']

            # plot all points
            for w in range(num_workers):
                a = plt.scatter(worker_arrays[w][:][:, 0], worker_arrays[w][:][:, 1], c=worker_arrays[w][:][:, 2],
                                marker='o', s=150)
                # plt.plot(worker_arrays[w][:, 0], worker_arrays[w][:, 1], '--', color='black', linewidth=1)
            cbar = plt.colorbar()
            # min_acc = (worker_arrays[:, :, 2]).min()
            # cbar.set_clim(min_acc, 100)
            # cbar.draw_all()
            s = plt.scatter(worker_arrays[:, 0, 0], worker_arrays[:, 0, 1], c='k', marker='2', s=150)
            e = plt.scatter(worker_arrays[:, -1, 0], worker_arrays[:, -1, 1], c='r', marker='+', s=150)
            plt.title("PBT test accuracy of all workers and epochs")
            plt.xlabel("{}".format(param_list[0]))
            plt.ylabel("{}".format(param_list[1]))
            plt.legend((s, e, a),
                       ('First epoch', 'Last epoch', 'All epochs'),
                       scatterpoints=1,
                       loc='lower right',
                       ncol=3,
                       fontsize=8)

            # save the figure to file
            path = os.path.join(train_dir, 'results.png')
            plt.savefig(path, dpi=500, bbox_inches='tight')
            print('visualization is saved to {}'.format(train_dir))
            # plt.show()
        else:
            print('number of varied parameters is bigger than 2, no plot can shown')
    else:
        print('no history table found')


def save_concat_imgs(images, img_names, save_path):
    '''
        Saves concatenated images to file
        Input:
        - images - list of np arrays of images (of length N each), which names are decribed in img_names
        - img_names - list if image names ('trg', 'ref', 'disp', 'depth', 'ref_inv_warp', 'ref_warped', 'trg-ref_warped')
        - filenames - list of N filenames
    '''
    imgs = [Image.fromarray(np.array(img_np*255).astype('uint8')) for img_np in images]
    # pick the image which is the smallest, and resize the others to match it
    min_shape = sorted([(np.sum(img.size), img.size) for img in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(img.resize(min_shape)) for img in imgs))

    # save combined image
    imgs_concat = PIL.Image.fromarray(imgs_comb)
    imgs_concat.save(save_path)





if __name__ == '__main__':
    # root_dir = '/media/victoria/d/models/mnist'
    # results_visualization(root_dir)

    train_dir = '/media/victoria/d/models/mnist/1812Dec17_16-23-11'
    num_workers = 10
    w_params_visualization(train_dir, num_workers)


























