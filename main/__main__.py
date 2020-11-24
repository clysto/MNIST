import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from main.tools import load_dataset
from tqdm import trange
from torch.autograd import Variable
from main.net import Net
from PIL import Image
import sys


def train():
    model = Net()
    print(model)

    X_train, Y_train = load_dataset(type="train")

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.00005, momentum=0.9)

    for i in (t := trange(10000)) :
        # 从样本集中随机抽取128个样本进行训练
        samp = np.random.randint(0, X_train.shape[0], size=(128))
        X = Variable(torch.tensor(X_train[samp].reshape((-1, 28 * 28))).float())
        Y = Variable(torch.tensor(Y_train[samp]).long())

        optimizer.zero_grad()
        out = model(X)

        loss = loss_function(out, Y)
        loss.backward()
        optimizer.step()

        cat = torch.argmax(out, dim=1)
        accuracy = (cat == Y).float().mean()
        loss = loss.mean()
        t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

    X_test, Y_test = load_dataset(type="test")
    Y_test_preds = torch.argmax(
        model(torch.tensor(X_test.reshape((-1, 28 * 28))).float()), dim=1
    ).numpy()
    r = (Y_test == Y_test_preds).mean()
    print(r)

    torch.save(model.state_dict(), "data/model.pt")


def extract_image():
    X, Y = load_dataset(type="test")

    for i in range(100):
        im = Image.fromarray(X[i])
        plt.imshow(im)
        im.save(f"data/extract/{i}-{Y[i]}.jpg")


if __name__ == "__main__":
    commad = sys.argv[1]
    if commad == "train":
        train()
    elif commad == "extract":
        extract_image()
