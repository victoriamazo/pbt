import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


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
            plt.colorbar()
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



























