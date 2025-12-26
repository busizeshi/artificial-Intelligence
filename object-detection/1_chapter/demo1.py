import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import time

# 加载预训练 Faster R-CNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO数据集的类别名称
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 打开视频文件
video_path = r'D:\cxx\artificial-Intelligence\data\object-detect\test.mp4'
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频的帧率和尺寸
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video FPS: {fps}, Width: {width}, Height: {height}")

# 初始化视频写入器（用于保存结果视频）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_detection_result.mp4', fourcc, fps, (width, height))

# 记录推理开始时间
start_time = time.time()

frame_count = 0

# 逐帧处理视频
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 将BGR格式的帧转换为RGB格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 转换为PIL图像
    pil_image = Image.fromarray(frame_rgb)
    
    # 预处理图像
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(pil_image)
    
    # 推理
    with torch.no_grad():
        predictions = model([img_tensor])
    
    # 设置置信度阈值
    confidence_threshold = 0.5
    scores = predictions[0]['scores']
    high_confidence_indices = scores > confidence_threshold
    
    boxes = predictions[0]['boxes'][high_confidence_indices]
    labels = predictions[0]['labels'][high_confidence_indices]
    
    # 创建图像副本用于绘制
    img_draw = pil_image.copy()
    img_draw = img_draw.convert("RGB")
    draw = ImageDraw.Draw(img_draw)

    # 绘制检测框和标签
    for box, label in zip(boxes, labels):
        box = [int(x) for x in box.tolist()]
        label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        
        # 绘制矩形框
        draw.rectangle(box, outline="red", width=2)
        
        # 添加标签文本（蓝色大字体）
        text = f"{label_name}"
        try:
            # 尝试使用一个较大的字体
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # 如果找不到字体文件，则使用默认字体
            font = ImageFont.load_default()
        
        # 获取文本尺寸
        if hasattr(draw, 'textbbox'):
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        elif hasattr(draw, 'textsize'):
            text_width, text_height = draw.textsize(text, font=font)
        else:
            # 如果没有上述方法，使用默认估算
            text_width = len(text) * 12
            text_height = 15
        
        # 绘制标签背景
        draw.rectangle([box[0], box[1]-text_height-10, box[0]+text_width+10, box[1]], fill="red")
        # 绘制蓝色大字体标签
        draw.text((box[0]+5, box[1]-text_height-5), text, fill="blue", font=font)
    
    # 将处理后的图像转换回numpy数组并转换为BGR格式
    result_frame = cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR)
    
    # 将结果帧写入输出视频
    out.write(result_frame)
    
    frame_count += 1

# 记录推理结束时间
end_time = time.time()
total_time = end_time - start_time
print(f"Total inference time: {total_time:.2f} seconds for {frame_count} frames")
print(f"Average time per frame: {total_time/frame_count:.4f} seconds")

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()