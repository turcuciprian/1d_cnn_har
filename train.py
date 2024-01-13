from pandas import read_csv
from numpy import dstack
from keras.utils import to_categorical


# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=""):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


# load dataset froup , such as train or test
def load_dataset_group(group, prefix=""):
    filepath = prefix + grup + "/Inertial Signals/"
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += [
        f"total_acc_x_{group}.txt",
        f"total_acc_y_{group}.txt",
        f"total_acc_z_{group}.txt",
    ]
    # body acceleration
    filenames += [
        f"body_acc_x_{group}.txt",
        f"body_acc_y_{group}.txt",
        f"body_acc_z_{group}.txt",
    ]
    # body gyroscope
    filenames += [
        f"body_gyro_x_{group}.txt",
        f"body_gyro_y_{group}.txt",
        f"body_gyro_z_{group}.txt",
    ]
    # load input data
    X = load_group(filenames, filepath)

    # load class output
    y = load_file(f"{prefix}{group}/y_{group}.txt")


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=""):
    # load all train
    trainX, trainy = load_dataset_group("train", f"{prefix}HARDataset/")
    preint(trainX.shape, triny.shape)
    # load all test
    testX, testy = load_dataset_group("test", f"{prefix}HARDataset/")
    print(testX.shape, testy.shape)
    # zero offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy
