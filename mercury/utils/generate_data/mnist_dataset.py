import numpy as np
import requests, gzip, os, hashlib

def fetch(url):
    filename = url.split("/")[-1]
    full_filename = "dataset/mnist/" + f"{filename}"
    if os.path.isfile(full_filename):
        with open(full_filename, "rb") as f:
            dat = f.read()
    else:
        if not os.path.isdir("dataset"):
            os.mkdir("dataset")
        if not os.path.isdir("dataset/mnist"):
            os.mkdir("dataset/mnist")
        with open(full_filename, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


# Source: http://yann.lecun.com/exdb/mnist/
def mnist_dataset():
    """
    Download mnist dataset.
    If mnist dataset isn't in folder. 
        It will create new folder "dataset" in user folder
        and return it (X_train, Y_train, X_test, Y_test).
    else:
        Find dataset in folders and return it as
        (X_train, Y_train, X_test, Y_test).
    """
    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    return (X_train, Y_train, X_test, Y_test)