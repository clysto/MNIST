import gzip
import numpy as np
from os import path
import glob
from PIL import Image
import main

project_fp = path.dirname(path.dirname(main.__file__))


def load_dataset(type="train"):
    """加载MNIST数据集"""
    if type == "train":
        images_fp = path.join(project_fp, "data/train-images-idx3-ubyte.gz")
        labels_fp = path.join(project_fp, "data/train-labels-idx1-ubyte.gz")
    elif type == "test":
        images_fp = path.join(project_fp, "data/t10k-images-idx3-ubyte.gz")
        labels_fp = path.join(project_fp, "data/t10k-labels-idx1-ubyte.gz")
    else:
        raise Exception("没有这个类型")

    with open(images_fp, "rb") as f:
        dat = f.read()
    X = (
        np.frombuffer(gzip.decompress(dat), dtype=np.uint8)
        .copy()[0x10:]
        .reshape((-1, 28, 28))
    )
    with open(labels_fp, "rb") as f:
        dat = f.read()
    Y = np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()[8:]
    return X, Y


def load_user_dataset():
    base_fp = path.join(project_fp, "data/generate")
    X = []
    Y = []
    for i in range(10):
        fps = glob.glob(path.join(base_fp, str(i), "*.jpg"))
        for fp in fps:
            im = Image.open(fp).convert("L")
            X.append(np.array(im))
            Y.append(i)
    return np.array(X), np.array(Y)
