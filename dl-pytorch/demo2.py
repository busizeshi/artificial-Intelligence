import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os


# ==========================================
# 1. 模型定义 (必须与训练时完全一致)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# ==========================================
# 2. 推理流程配置
# ==========================================
def load_model(model_path, device):
    """
    加载训练好的模型参数
    """
    model = SimpleCNN().to(device)

    if os.path.exists(model_path):
        # map_location 确保在 CPU 上也能加载 GPU 训练的模型
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")
    else:
        print(f"警告: 未找到模型文件 {model_path}。将使用未训练的随机参数模型进行演示。")
        print("请先运行训练脚本 (mnist_tutorial.py) 生成模型文件。")

    # 关键步骤：切换到评估模式！
    # 这会固定 Dropout 和 BatchNorm，保证推理结果稳定
    model.eval()
    return model


def predict_single_image(model, image_tensor, device):
    """
    对单张图片进行推理
    """
    # 1. 处理数据维度
    # 输入图片通常是 (C, H, W) 或 (H, W)，但模型需要 (Batch, C, H, W)
    # 所以我们需要用 unsqueeze(0) 增加一个 batch 维度
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # 2. 不计算梯度 (节省资源)
    with torch.no_grad():
        output = model(image_tensor)

        # 3. 获取预测结果
        # output 是对数概率，exp(output) 得到真实概率
        probabilities = torch.exp(output)

        # 获取概率最大的类别
        prediction = output.argmax(dim=1, keepdim=True).item()
        confidence = probabilities[0][prediction].item()

    return prediction, confidence


# ==========================================
# 3. 模拟真实场景
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径指向上一节保存的模型
    model_path = "../data/model_wight/mnist_cnn.pt"

    # 加载模型
    model = load_model(model_path, device)

    # --- 获取一张测试图片 ---
    # 在实际应用中，这里可能是读取用户上传的 png/jpg 文件
    # 这里我们从测试集中随机取一张
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 随机选择索引
    idx = torch.randint(0, len(test_dataset), (1,)).item()
    image_tensor, label = test_dataset[idx]

    # --- 进行推理 ---
    pred_label, confidence = predict_single_image(model, image_tensor, device)

    # --- 可视化结果 ---
    print(f"\n图片索引: {idx}")
    print(f"真实标签 (Ground Truth): {label}")
    print(f"模型预测 (Prediction): {pred_label}")
    print(f"置信度 (Confidence): {confidence:.4f}")

    # 绘图
    plt.figure(figsize=(4, 4))
    plt.imshow(image_tensor.squeeze(), cmap='gray')
    plt.title(f"True: {label} | Pred: {pred_label} ({confidence:.1%})",
              color=("green" if label == pred_label else "red"))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()