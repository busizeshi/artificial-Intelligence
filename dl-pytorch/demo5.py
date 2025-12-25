"""
模型推理
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 环境配置
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './flower_model.pth'
json_path = 'D:/cxx/artificial-Intelligence/data/cat_to_name.json'
# 替换为你想要测试的图片路径
image_path = 'D:/cxx/artificial-Intelligence/data/flower_data/flower.png'

# 加载类别映射
with open(json_path, 'r') as f:
    cat_to_name = json.load(f)


# ==========================================
# 2. 加载模型结构与权重
# ==========================================
def load_trained_model(checkpoint_path, num_classes=102):
    # 必须重建与训练时一模一样的结构
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 检查 checkpoint 是否为包含 state_dict 的字典（而不是直接的 state_dict）
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            # 如果 checkpoint 包含 'state_dict' 键，则使用它
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 如果 checkpoint 本身就是 state_dict，则直接使用
            model.load_state_dict(checkpoint)
    else:
        # 如果 checkpoint 是 state_dict 直接加载
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()  # 极其重要：进入推理模式
    return model


model = load_trained_model(model_path)


# ==========================================
# 3. 图像预处理 (必须与验证集一致)
# ==========================================
def process_image(image_path):
    img = Image.open(image_path).convert('RGB')

    # 定义预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_tensor = preprocess(img)
    # 扩展维度：从 [C, H, W] 变成 [1, C, H, W] (增加 Batch 维度)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


# ==========================================
# 4. 执行推理
# ==========================================
img_tensor = process_image(image_path).to(device)

with torch.no_grad():  # 推理时不需要计算梯度
    output = model(img_tensor)
    # 将输出转化为概率
    probs = torch.nn.functional.softmax(output, dim=1)
    # 找到概率最大的类别
    top_prob, top_class = torch.max(probs, 1)

# 获取类别名称
class_idx = str(top_class.item() + 1)  # 注意：ImageFolder 的索引通常从0开始，而JSON可能从1开始，需根据实际数据调整
flower_name = cat_to_name.get(class_idx, "Unknown")

print(f"预测结果: {flower_name}")
print(f"置信度: {top_prob.item() * 100:.2f}%")

# ==========================================
# 5. 可视化展示
# ==========================================
plt.imshow(Image.open(image_path))
plt.title(f"Prediction: {flower_name} ({top_prob.item() * 100:.2f}%)")
plt.axis('off')
plt.show()