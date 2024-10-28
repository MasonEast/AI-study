from .main import MNISTNN, device, myNN
import torch

import torchvision.io
from torchvision.io import ImageReadMode

def loadImg(path):
    return 1. - torchvision.io.read_image(path, mode=ImageReadMode.GRAY).to(device=device) / 255.
imgs = [(0, loadImg('0.png')),
        (1, loadImg('1.png')),
        (5, loadImg('5.png')),
        (7, loadImg('7.png')),
        (8, loadImg('8.png')),
        (9, loadImg('9.png')),]
with torch.no_grad():
    for num, img in imgs:
        output = myNN(img)
        print(f'实际值: {num}，预测值：{output.argmax(dim=1).item()}')