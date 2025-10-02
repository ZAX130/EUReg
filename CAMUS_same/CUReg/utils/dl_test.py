import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

# print("torch.cuda.nccl.version(): {}".format(torch.cuda.nccl.version()))

# # 设置第二张显卡为当前设备
# device_index = 1  # 第二张显卡的索引为1
# torch.cuda.set_device(device_index)

# 输出当前CUDA设备索引
print(f"Current CUDA device index: {torch.cuda.current_device()}")

# 定义一个简单的卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train():
    # 加载CIFAR-10数据集并进行预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    # 初始化模型、损失函数和优化器
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 在GPU上进行训练（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:  # 检查是否有多个GPU可用
        print("使用", torch.cuda.device_count(), "个GPU")
        net = nn.DataParallel(net)  # 使用DataParallel将模型包装起来
    net.to(device)

    # 训练网络
    for epoch in range(5):  # 进行五个训练轮次
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # 每100个小批量数据输出一次损失值
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Training finished.")

if __name__ == '__main__':
    train()