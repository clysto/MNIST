import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from main.tools import load_dataset, load_user_dataset
from tqdm import trange
from torch.autograd import Variable
from main.net import Net
from PIL import Image
import sys
import os


def train(is_user):
    model = Net()
    print(model)

    if is_user:
        X_train, Y_train = load_user_dataset()
    else:
        X_train, Y_train = load_dataset(type="train")

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.00005, momentum=0.9)

    for _ in (t := trange(2000)) :
        # 从样本集中随机抽取128个样本进行训练
        samp = np.random.randint(0, X_train.shape[0], size=(128))
        X = Variable(torch.unsqueeze(torch.tensor(X_train[samp]).float(), 1))
        Y = Variable(torch.tensor(Y_train[samp]).long())

        # 正向传播
        out = model(X)
        loss = loss_function(out, Y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cat = torch.argmax(out, dim=1)
        accuracy = (cat == Y).float().mean()
        loss = loss.mean()
        t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

    X_test, Y_test = load_dataset(type="test")
    Y_test_preds = torch.argmax(
        model(torch.unsqueeze(torch.tensor(X_test).float(), 1)), dim=1
    ).numpy()
    r = (Y_test == Y_test_preds).mean()
    print(r)

    torch.save(model.state_dict(), "data/model.pt")


def extract_image():
    X, Y = load_dataset(type="test")

    if not os.path.isdir("data/extract"):
        os.mkdir("data/extract")

    for i in range(100):
        im = Image.fromarray(X[i])
        plt.imshow(im)
        im.save(f"data/extract/{i}-{Y[i]}.jpg")


if __name__ == "__main__":
    commad = sys.argv[1]
    if commad == "train":
        train(False)
    elif commad == "train_user":
        train(True)
    elif commad == "extract":
        extract_image()
