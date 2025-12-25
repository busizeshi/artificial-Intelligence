import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np


# ==========================================
# 1. 数据集定义 (Dataset)
# ==========================================
class FlowerDataset(Dataset):
    """
    自定义数据集，支持从标注文件读取
    """

    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_names, self.labels = self._load_annotations(annotations_file)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 拼接完整路径
        img_path = os.path.join(self.root_dir, self.img_names[idx])

        # 读取图片并转换为 RGB (防止出现RGBA或灰度图导致维度错误)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个空白图像或跳过（此处简化处理）
            image = Image.new('RGB', (64, 64))

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # 标签转为 long 类型张量，分类任务的标准要求
        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def _load_annotations(self, annotations_file):
        img_names = []
        labels = []
        with open(annotations_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 2:
                    img_names.append(parts[0])
                    labels.append(int(parts[1]))
        return img_names, labels


# ==========================================
# 2. 配置参数与数据增强
# ==========================================
CONFIG = {
    'data_dir': '../data/flower_data2',
    'batch_size': 64,
    'num_epochs': 25,
    'lr': 1e-3,
    'input_size': 224,  # ResNet18 推荐输入 224
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(CONFIG['input_size']),  # 随机裁剪到目标尺寸
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(CONFIG['input_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ==========================================
# 3. 实例化 DataLoaders
# ==========================================
train_dataset = FlowerDataset(
    root_dir=os.path.join(CONFIG['data_dir'], 'train_filelist'),
    annotations_file=os.path.join(CONFIG['data_dir'], 'train.txt'),
    transform=data_transforms['train']
)

valid_dataset = FlowerDataset(
    root_dir=os.path.join(CONFIG['data_dir'], 'val_filelist'),
    annotations_file=os.path.join(CONFIG['data_dir'], 'val.txt'),
    transform=data_transforms['valid']
)

# 获取分类总数 (动态获取，而不是硬编码 102)
num_classes = len(set(train_dataset.labels))
print(f"Detected {num_classes} classes.")

# 性能优化：num_workers 加速数据读取, pin_memory 加速数据传输到 GPU
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True),
    'valid': DataLoader(valid_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
}


# ==========================================
# 4. 模型初始化
# ==========================================
def initialize_model(num_classes, feature_extract=True):
    model = models.resnet18(pretrained=True)

    # 如果是特征提取模式，冻结模型参数
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    # 修改最后的全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


model_ft = initialize_model(num_classes).to(CONFIG['device'])

# 优化器只针对需要更新的参数
params_to_update = [p for p in model_ft.parameters() if p.requires_grad]
optimizer = optim.Adam(params_to_update, lr=CONFIG['lr'])

# 修改：对于 StepLR，它根据 step 计数衰减，而不是根据 Loss
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()


# ==========================================
# 5. 训练函数
# ==========================================
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
                inputs = inputs.to(CONFIG['device'])
                labels = labels.to(CONFIG['device'])

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

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度学习最佳实践：在验证集表现最好时保存权重
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'best_flower_model.pth')

        # 更新学习率调度器 (StepLR 是按 epoch 走的)
        scheduler.step()
        print(f"Next Learning Rate: {optimizer.param_groups[0]['lr']}")
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model


# 开始训练
if __name__ == "__main__":
    trained_model = train_model(model_ft, dataloaders, criterion, optimizer, scheduler, CONFIG['num_epochs'])