import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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

class MNISTNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), # 我们首先需要将二维的图片压平
            # 剩下的部分其实很好理解了
            nn.Linear(28 * 28, 512),  # MNIST是一个28*28的黑白图片，我们读取灰度值就行了。上面我们已经将其压平了，剩下28*28长度的张量，代表每个像素的灰度值
            nn.GELU(), # GELU是ReLU激活函数的增强版，其作用是在输出结果为负数的时候有一定的弧度，而不像ReLU一样直接为0，能够更好帮助我们完成梯度下降
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        return self.network(x)

def train(*, model, data_loader, loss_fn, optimizer):
    size = len(data_loader.dataset)
    model.train()
    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
       
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() # 将梯度清零，防止因为梯度累积造成优化的时候用力过猛
       
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
           
def test(*, model, data_loader, loss_fn, optimizer):
        size = len(data_loader.dataset)
        num_batches = len(data_loader)
        model.eval()
        test_loss, correct = 0, 0
       
        with torch.no_grad():
            count = 0.
            for (x, y) in data_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_loss += loss_fn(pred, y).item()
                correct += pred.argmax(1).eq(y).type(torch.float).sum().item()
            # 因为我们的输出的是10个概率值，取最大的那个对应的标签，其中的“1”代表取维度1的那个最大值
            print(f'Test loss: {test_loss/num_batches}')
            print(f"Test Accuracy: {100*correct/size}%")

myNN = MNISTNN().to(device)
opti = torch.optim.Adam(myNN.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train(model=myNN, data_loader=training_dataLoader, loss_fn=loss_fn, optimizer=opti)
    test(model=myNN, data_loader=test_dataLoader, loss_fn=loss_fn, optimizer=opti)
