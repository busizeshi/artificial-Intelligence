import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ==========================================
# 1. 基础配置与设备选择 (Configuration)
# ==========================================
# 检查是否有 GPU (CUDA) 或 Mac 的 MPS 加速，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"当前使用的计算设备: {device}")

# 超参数设置
BATCH_SIZE = 64  # 每次喂给模型的数据量
LEARNING_RATE = 0.01  # 学习率，控制参数更新的步长
EPOCHS = 3  # 训练总轮数 (为了演示快速运行，设为3，实际可设为10以上)


# ==========================================
# 2. 数据准备 (Data Preparation)
# ==========================================
def get_data_loaders():
    """
    下载并加载 MNIST 数据集，进行预处理
    """
    # 定义数据转换管道
    # ToTensor: 将图片(H, W, C)转换为Tensor (C, H, W)并将像素值归一化到[0, 1]
    # Normalize: 标准化 (mean=0.1307, std=0.3081 是MNIST数据集计算出的经验值)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载/加载训练集
    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    # 下载/加载测试集
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    print(type(train_dataset))

    # 创建 DataLoader，用于批次读取数据
    # shuffle=True 表示每个 epoch 都会打乱数据，这对训练很重要
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


# ==========================================
# 3. 定义神经网络模型 (Model Definition)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层 1: 输入通道1 (灰度图), 输出通道32, 卷积核3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 卷积层 2: 输入通道32, 输出通道64, 卷积核3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 丢弃层: 防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层 1: 输入特征数需计算 (64 * 7 * 7), 输出 128
        # 经过两次2x2最大池化(MaxPooling)，28x28 的图片会变成 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层 2 (输出层): 输出 10 (对应数字 0-9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x shape: [Batch, 1, 28, 28]
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 图片变 14x14

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 图片变 7x7

        x = self.dropout1(x)

        # 展平 Tensor: [Batch, 64, 7, 7] -> [Batch, 64*7*7]
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)

        # 输出层使用 LogSoftmax，配合 NLLLoss 使用（或者直接输出用CrossEntropyLoss）
        output = F.log_softmax(x, dim=1)
        return output


# ==========================================
# 4. 训练函数 (Training Loop)
# ==========================================
def train(model, device, train_loader, optimizer, epoch):
    model.train()  # 切换到训练模式 (启用 Dropout 和 BatchNorm)
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移动到 GPU 或 CPU
        data, target = data.to(device), target.to(device)

        # 1. 梯度清零 (PyTorch 累积梯度，所以每次反向传播前必须清零)
        optimizer.zero_grad()

        # 2. 前向传播 (计算预测值)
        output = model(data)

        # 3. 计算损失 (负对数似然损失，对应 LogSoftmax)
        loss = F.nll_loss(output, target)

        # 4. 反向传播 (计算梯度)
        loss.backward()

        # 5. 参数更新
        optimizer.step()

        # 打印训练进度
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# ==========================================
# 5. 验证/评估函数 (Evaluation)
# ==========================================
def test(model, device, test_loader):
    model.eval()  # 切换到评估模式 (关闭 Dropout 等)
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 不计算梯度，节省显存并加速
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # 获取预测结果 (概率最大的索引)
            pred = output.argmax(dim=1, keepdim=True)

            # 统计预测正确的数量
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')


# ==========================================
# 6. 可视化预测结果 (Visualization)
# ==========================================
def visualize_predictions(model, device, test_loader):
    print("正在生成可视化预测结果...")
    model.eval()

    # 获取一个批次的数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    output = model(images)
    preds = output.argmax(dim=1, keepdim=True)

    # 绘图配置
    figure = plt.figure(figsize=(10, 5))
    cols, rows = 5, 2

    # 随机展示 10 张图
    for i in range(1, cols * rows + 1):
        # 移回 CPU 进行绘图
        img = images[i - 1].cpu().squeeze()
        label = labels[i - 1].item()
        pred = preds[i - 1].item()

        figure.add_subplot(rows, cols, i)
        plt.title(f"Label: {label}\nPred: {pred}", color=("green" if label == pred else "red"))
        plt.axis("off")
        plt.imshow(img, cmap="gray")

    plt.tight_layout()
    plt.show()


# ==========================================
# 主程序入口
# ==========================================
if __name__ == '__main__':
    # 1. 获取数据
    train_loader, test_loader = get_data_loaders()

    # 2. 初始化模型
    model = SimpleCNN().to(device)

    # 3. 定义优化器 (SGD 随机梯度下降)
    optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)

    # 4. 循环训练和验证
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # 5. 保存模型 (可选)
    torch.save(model.state_dict(), "../data/model_wight/mnist_cnn.pt")
    print("模型已保存为 mnist_cnn.pt")

    # 6. 可视化结果
    visualize_predictions(model, device, test_loader)