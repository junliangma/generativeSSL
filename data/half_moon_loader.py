from sklearn.datasets import make_moons
import numpy as np
import pdb

def _download():
    train_x, train_t = make_moons(n_samples=10000, shuffle=True, noise=0.2, random_state=1234)
    test_x, test_t = make_moons(n_samples=10000, shuffle=True, noise=0.2, random_state=1234)
    valid_x, valid_t = make_moons(n_samples=10000, shuffle=True, noise=0.2, random_state=1234)

    train_x += np.abs(train_x.min())
    test_x += np.abs(test_x.min())
    valid_x += np.abs(valid_x.min())

    train_set = (train_x.astype('float32'), train_t.astype('int32'))
    test_set = (test_x.astype('float32'), test_t.astype('int32'))
    valid_set = (valid_x.astype('float32'), valid_t.astype('int32'))
    return train_set, test_set, valid_set

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

def load_semi_supervised():
    """
    Load the half moon dataset with 6 fixed labeled data points.
    """

    train_set, test_set, valid_set = _download()

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
