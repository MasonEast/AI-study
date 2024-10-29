import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

training_data = datasets.MNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=ToTensor(), download=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 如果是N卡，则使用CUDA进行加速
device = 'mps' if torch.backends.mps.is_available() else device # 如果是Mac，使用Metal Performance Shader进行加速
# 如果都不是，使用CPU

print(device)

torch.set_default_device(device) # 设置默认设备

batch_size = 100

# torch.Generator(device=device)作用是将数据加载到指定的设备上
training_dataLoader = DataLoader(training_data, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
test_dataLoader = DataLoader(test_data, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) # 1表示输入通道数，32表示输出通道数，3表示卷积核大小，1表示步长，padding表示填充
        self.bn1 = nn.BatchNorm2d(32) # 归一化层
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # kernel_size表示池化窗口大小，stride表示步长，padding表示填充
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # 应用第一个卷积层，后接GELU激活和池化层
        x = self.pool(F.gelu(self.bn1(self.conv1(x))))
        
        # 应用第二个卷积层，后接GELU激活和池化层
        x = self.pool(F.gelu(self.bn2(self.conv2(x))))
        
        # 这里打印张量形状以确定尺寸是正确的

        # 展平操作
        x = x.view(x.shape[0], -1)

        # print(f'Shape after flatten: {x.shape}')
        
        # 应用全连接层和激活函数
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)

        return x
    
def train(*, model, data_loader, loss_fn, optimizer):
    size = len(data_loader.dataset) # 获取数据集的大小
    model.train() # 将模型设置为训练模式
    for batch, (x, y) in enumerate(data_loader): # 遍历数据集
        x, y = x.to(device), y.to(device) # 将数据集放到GPU（CPU）上
        pred = model(x) # 前向传播，计算预测值pred。调用model(x)会应用模型的当前参数去进行计算
        loss = loss_fn(pred, y) # 计算损失
       
        optimizer.zero_grad() # 将梯度清零，防止因为梯度累积造成优化的时候用力过猛

        loss.backward() # 进行反向传播，计算损失相对于模型参数的梯度。这个过程是自动进行的，由PyTorch根据前向传播计算图自动管理梯度计算。

        optimizer.step() # 根据计算出的梯度对模型参数进行调整，以尽量减小损失。
       
        if batch % 100 == 0:   # 每训练100个batch，打印一次损失
            loss, current = loss.item(), batch * len(x)    # 计算当前损失
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
           
def test(*, model, data_loader, loss_fn, optimizer):
        size = len(data_loader.dataset)
        num_batches = len(data_loader)
        model.eval() # 将模型设置为评估模式
        test_loss, correct = 0, 0
       
        with torch.no_grad(): # 不计算梯度，节省内存
            for (x, y) in data_loader: # 遍历数据集
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_loss += loss_fn(pred, y).item() # sum up batch loss
                correct += pred.argmax(1).eq(y).type(torch.float).sum().item()  # 因为我们的输出的是10个概率值，取最大的那个对应的标签，其中的“1”代表取维度1的那个最大值
           
            print(f'Test loss: {test_loss/num_batches}')
            print(f"Test Accuracy: {100*correct/size}%")

if __name__ == '__main__':

    myCNN = MNISTCNN().to(device) # 将模型放到GPU上
    opti = torch.optim.Adam(myCNN.parameters(), lr=1e-3) # 使用adam函数更新参数
    loss_fn = nn.CrossEntropyLoss() # 使用交叉熵损失函数
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train(model=myCNN, data_loader=training_dataLoader, loss_fn=loss_fn, optimizer=opti)
        test(model=myCNN, data_loader=test_dataLoader, loss_fn=loss_fn, optimizer=opti)

    # 保存模型的状态字典， 推荐使用第一种方法。虽然需要在加载时重新定义模型结构，但它更加灵活，并避免了与特定 PyTorch 版本的绑定问题。只保存模型的权重通常也可以实现更大的兼容性和重新训练的能力。
    torch.save(myCNN.state_dict(), 'model.pth')

    # 保存整个模型
    torch.save(myCNN, 'model_complete.pth')