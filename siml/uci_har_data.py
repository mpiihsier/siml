import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


def read_signals(filename):
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.strip().split(), data)
        data = [list(map(float, line)) for line in data]
    return data

def read_labels(filename):        
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(lambda x: int(x)-1, activities))
    return activities

def load():
    DATA_FOLDER = '../datasets/UCI HAR Dataset/'
    INPUT_FOLDER_TRAIN = DATA_FOLDER+'train/Inertial Signals/'
    INPUT_FOLDER_TEST = DATA_FOLDER+'test/Inertial Signals/'

    INPUT_FILES_TRAIN = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt', 
                             'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
                                                 'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']

    INPUT_FILES_TEST = ['body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt', 
                            'body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt',
                                                 'total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt']

    LABELFILE_TRAIN = DATA_FOLDER+'train/y_train.txt'
    LABELFILE_TEST = DATA_FOLDER+'test/y_test.txt'

    train_signals, test_signals = [], []

    for input_file in INPUT_FILES_TRAIN:
        signal = read_signals(INPUT_FOLDER_TRAIN + input_file)
        train_signals.append(signal)
    train_signals = np.transpose(train_signals, (1, 2, 0))

    for input_file in INPUT_FILES_TEST:
        signal = read_signals(INPUT_FOLDER_TEST + input_file)
        test_signals.append(signal)
    test_signals = np.transpose(test_signals, (1, 2, 0))

    train_labels = read_labels(LABELFILE_TRAIN)
    test_labels = read_labels(LABELFILE_TEST)

    [no_signals_train, no_steps_train, no_components_train] = np.shape(train_signals)
    [no_signals_test, no_steps_test, no_components_test] = np.shape(test_signals)
    no_labels = len(np.unique(train_labels[:]))

    print("The train dataset contains {} signals, each one of length {} and {} components ".format(no_signals_train, no_steps_train, no_components_train))
    print("The test dataset contains {} signals, each one of length {} and {} components ".format(no_signals_test, no_steps_test, no_components_test))
    print("The train dataset contains {} labels, with the following distribution:\n {}".format(np.shape(train_labels)[0], Counter(train_labels[:])))
    print("The test dataset contains {} labels, with the following distribution:\n {}".format(np.shape(test_labels)[0], Counter(test_labels[:])))

    return train_signals, train_labels, test_signals, test_labels

