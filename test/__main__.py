from main.net import Net
import torch
from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model = Net()
    model.load_state_dict(torch.load("data/model.pt"))
    model.eval()
    fp = sys.argv[1]
    im = Image.open(fp).convert("L")
    x = np.array(im)
    y = model(torch.tensor([x]).reshape((-1, 28 * 28)).float())
    print((torch.argmax(y, dim=1)).item())
