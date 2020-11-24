import gzip
import numpy as np


def load_dataset(type="train"):
    """加载MNIST数据集"""
    if type == "train":
        images_fp = "data/train-images-idx3-ubyte.gz"
        labels_fp = "data/train-labels-idx1-ubyte.gz"
    elif type == "test":
        images_fp = "data/t10k-images-idx3-ubyte.gz"
        labels_fp = "data/t10k-labels-idx1-ubyte.gz"
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
