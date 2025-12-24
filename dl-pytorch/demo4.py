import os
import time
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ==========================================
# 1. 配置参数
# ==========================================
data_dir = '../data/flower_data/'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
# 修改这里：填入你 cat_to_name.json 的绝对路径
json_path = r'D:\cxx\artificial-Intelligence\data\cat_to_name.json'

batch_size = 32  # 建议在个人电脑上先用较小的 batch_size
num_epochs = 20
model_name = 'resnet'
num_classes = 102  # 根据实际分类数量修改
feature_extract = True  # 初步训练时冻结卷积层，只训练全连接层

# 检测设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==========================================
# 2. 数据预处理与加载
# ==========================================
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # ResNet 标准输入通常是 224x224
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集
try:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                   for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
except Exception as e:
    print(f"错误: 无法加载数据。请检查路径 {data_dir} 是否正确。详细信息: {e}")
    exit()

# 读取分类映射 (使用配置的 json_path)
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
else:
    print(f"警告: 未找到分类映射文件 {json_path}，将使用默认类别索引。")
    cat_to_name = {str(i): name for i, name in enumerate(class_names)}


# ==========================================
# 3. 模型初始化
# ==========================================
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # 使用 ResNet-18
    model_ft = models.resnet18(pretrained=use_pretrained)
    # 是否冻结卷积层参数
    set_parameter_requires_grad(model_ft, feature_extract)

    # 修改最后的全连接层，适应新的分类任务
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft


model_ft = initialize_model(num_classes, feature_extract)
model_ft = model_ft.to(device)

# 收集需要训练的参数
params_to_update = []
for name, param in model_ft.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)

# 定义优化器和损失函数
optimizer_ft = optim.Adam(params_to_update, lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# ==========================================
# 4. 训练函数
# ==========================================
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'valid':
                val_acc_history.append(epoch_acc)
            else:
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print(f'训练完成，用时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证准确率: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model


# ==========================================
# 5. 执行训练
# ==========================================
if __name__ == '__main__':
    # 第一阶段：只训练输出层
    print("开始第一阶段训练（冻结特征提取层）...")
    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, scheduler, num_epochs=10)

    # 第二阶段：解冻所有层，进行微调
    print("\n开始第二阶段训练（解冻全连接层并微调）...")
    for param in model_ft.parameters():
        param.requires_grad = True

    # 使用较小的学习率
    optimizer_fine_tune = optim.Adam(model_ft.parameters(), lr=1e-4)
    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_fine_tune, scheduler, num_epochs=10)

    # 保存模型
    torch.save(model_ft.state_dict(), 'flower_model.pth')
    print("模型已保存为 flower_model.pth")