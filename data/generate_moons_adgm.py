from sklearn.datasets import make_moons
import numpy as np
import pdb, pickle, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc


def pad_targets(xy):
    """
    Pad the targets to be 1hot.
    :param xy: A tuple containing the x and y matrices.
    :return: The 1hot coded dataset.
    """
    x, y = xy
    classes = np.max(y) + 1
    tmp_data_y = np.zeros((x.shape[0], classes))
    for i, dp in zip(range(len(y)), y):
        r = np.zeros(classes)
        r[dp] = 1
        tmp_data_y[i] = r
    y = tmp_data_y
    return x, y


def _download(noise):
    train_x, train_t = make_moons(n_samples=10000, shuffle=True, noise=noise, random_state=1234)
    test_x, test_t = make_moons(n_samples=10000, shuffle=True, noise=noise, random_state=1234)
    valid_x, valid_t = make_moons(n_samples=10000, shuffle=True, noise=noise, random_state=1234)

    train_x += np.abs(train_x.min())
    test_x += np.abs(test_x.min())
    valid_x += np.abs(valid_x.min())

    train_set = (train_x, train_t)
    test_set = (test_x, test_t)
    valid_set = (valid_x, valid_t)

    return train_set, test_set, valid_set


def load_semi_supervised(noise=0.2):
    """
    Load the half moon dataset with 6 fixed labeled data points.
    """

    train_set, test_set, valid_set = _download(noise)

    # Add 6 static labels.
    train_x_l = np.zeros((6, 2))
    train_t_l = np.array([0, 0, 0, 1, 1, 1])
    # Top halfmoon
    train_x_l[0] = [.7, 1.7]  # left
    train_x_l[1] = [1.6, 2.6]  # middle
    train_x_l[2] = [2.7, 1.7]  # right

    # Bottom halfmoon
    train_x_l[3] = [1.6, 2.0]  # left
    train_x_l[4] = [2.7, 1.1]  # middle
    train_x_l[5] = [3.5, 2.0]  # right
    train_set_labeled = (train_x_l, train_t_l)

    train_set_labeled = pad_targets(train_set_labeled)
    train_set = pad_targets(train_set)
    test_set = pad_targets(test_set)
    if valid_set is not None:
        valid_set = pad_targets(valid_set)

    return train_set, train_set_labeled, test_set, valid_set


## argv[1] - noise level

if __name__ == "__main__":
    noise = float(sys.argv[1])
    train, labeled, test, valid = load_semi_supervised(noise)
    data = {}
    data['x'], data['y'] = train[0], train[1]
    data['x_labeled'], data['y_labeled'] = labeled[0], labeled[1]
    data['x_test'], data['y_test'] = test[0], test[1]

    target = './data/moons_semi_'+str(noise)+'.pkl'
    with open(target, 'wb') as f:
	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
   
    plt.figure()
    x0 = data['x'][np.where(data['y'][:,0]==1)]
    x1 = data['x'][np.where(data['y'][:,1]==1)]
    plt.scatter(x0[:,0], x0[:,1], color='r', s=1)
    plt.scatter(x1[:,0], x1[:,1], color='b', s=1)
    xlabeled = data['x_labeled']
    plt.scatter(xlabeled[:,0], xlabeled[:,1], color='black', s=20)	
    plt.savefig('./data/moons_plot_semi', bbox_inches='tight') 
