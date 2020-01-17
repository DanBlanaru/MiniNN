import numpy as np


def standard_scaler(dataset):
    dataset = np.array(dataset)
    means = dataset.mean(axis=0)
    stds = dataset.std(axis=0)
    return (dataset - means) / stds


def min_max_scaler(dataset):
    dataset = np.array(dataset)
    maxs = dataset.max(axis=0)
    mins = dataset.min(axis=0)
    return (dataset - mins) / (maxs - mins)


def make_minibatches(dataset, target, minibatch_size):
    x = np.array(dataset)
    y = np.array(target)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]

    full_minibatches = x.shape[0] // minibatch_size

    batches = [(x[i * minibatch_size: (i + 1) * minibatch_size],
                y[i * minibatch_size: (i + 1) * minibatch_size])
               for i in range(full_minibatches)]

    if x.shape[0] % minibatch_size:
        batches.append((x[minibatch_size * full_minibatches:], y[minibatch_size * full_minibatches:]))

    return batches


def make_onehot_2d(true_classes, total_classes):
    y = np.zeros((len(true_classes), total_classes))
    for i, true_class in enumerate(true_classes):
        y[i][true_class] = 1
    return y
