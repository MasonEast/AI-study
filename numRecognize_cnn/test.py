
import torch

import os
import torchvision.io
from torchvision.io import ImageReadMode

from main import MNISTCNN

base_directory = r'C:\Users\admin\Desktop\ai\AI-study\numRecognize_cnn'   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 如果是N卡，则使用CUDA进行加速
device = 'mps' if torch.backends.mps.is_available() else device # 如果是Mac，使用Metal Performance Shader进行加速

def loadImg(path):
    return 1. - torchvision.io.read_image(os.path.join(base_directory, 'img', path), mode=ImageReadMode.GRAY).to(device=device) / 255.

imgs = [(0, loadImg('0.png')),
        (1, loadImg('1.png')),
        (3, loadImg('3.png')),
        (4, loadImg('4.png')),
        (5, loadImg('5.png')),
        (7, loadImg('7.png')),
        (8, loadImg('8.png')),
        (9, loadImg('9.png')),]


model = MNISTCNN().to(device)
# 加载整个模型
model = torch.load(os.path.join(base_directory, 'model_complete.pth'))

# 将模型设置为评估模式
model.eval()

print ('模型加载完成')
# 使用该模型进行推理
with torch.no_grad():

    for num, img in imgs:
        
        # 使用 unsqueeze 添加批量维度
        img = img.unsqueeze(0)
        print(f'img shape: {img.shape}')

        output = model(img)
        print(f'实际值: {num}，预测值：{output.argmax(dim=1).item()}')