# 使用pytorch神经网络进行气温预测
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings
import datetime
from sklearn import preprocessing

features = pd.read_csv("../data/temperature.csv")

print(features.head())
print("数据维度", features.shape)

years = features["year"]
months = features["month"]
days = features["day"]

# datetime格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]
print(dates[:5])

plt.style.use('fivethirtyeight')

fig,((ax1,ax2),(ax3,ax4))=plt.subplots(nrows=2, ncols=2, figsize=(10,10))
fig.autofmt_xdate(rotation=45)

ax1.plot(dates,features['actual'])
ax1.set_xlabel('Date')
ax1.set_ylabel('Temperature (Fahrenheit)')
ax1.set_title('Actual Maximum Temperature')

ax2.plot(dates,features['temp_1'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Temperature (Fahrenheit)')
ax2.set_title('Previous Maximum Temperature')

ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date')
ax3.set_ylabel('Temperature')
ax3.set_title('Two Days Prior Max Temp')

ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date')
ax4.set_ylabel('Temperature')
ax4.set_title('Friend Estimate')

plt.tight_layout(pad=2)
# plt.show()

# 独热编码
features=pd.get_dummies(features)   # one-hot
print(features.head())

labels=np.array(features['actual'])
features=features.drop('actual',axis=1)
feature_list=list(features.columns)
features=np.array(features)
print(features.shape)

input_features=preprocessing.StandardScaler().fit_transform(features)
print(input_features[0])

# 构建网络模型
x=torch.tensor(input_features,dtype=torch.float)
y=torch.tensor(labels,dtype=torch.float)

# 权重参数初始化
weights=torch.randn(14,128,dtype=torch.float,requires_grad=True)
biases=torch.randn(128,dtype=torch.float,requires_grad=True)
weights2=torch.randn(128,1,dtype=torch.float,requires_grad=True)
biases2=torch.randn(1,dtype=torch.float,requires_grad=True)

learning_rate=0.001
losses=[]

for i in range(1000):
    # 计算隐层
    hidden=x.mm(weights)+biases
    # 加入激活函数
    hidden=torch.relu( hidden)
    # 预测结果
    predictions=hidden.mm(weights2)+biases2
    # 计算损失
    loss=torch.mean((predictions-y)**2)
    losses.append(loss.data.numpy())

    if i%100==0:
        print(f"第{i}次迭代，损失为{loss}")
    # 反向传播
    loss.backward()

    # 更新参数
    weights.data.add_(-learning_rate*weights.grad.data)
    biases.data.add_(-learning_rate*biases.grad.data)
    weights2.data.add_(-learning_rate*weights2.grad.data)
    biases2.data.add_(-learning_rate*biases2.grad.data)

    # 每次迭代都得清空上次梯度
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()

print('----------------------'*5)

# 更简单的构建网络模型
input_size=input_features.shape[1]
hidden_size=128
output_size=1
batch_size=16

model=torch.nn.Sequential(
    torch.nn.Linear(input_size,hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size,output_size)
)

cost=torch.nn.MSELoss(reduction='mean')
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

# 训练网络
losses=[]
for i in range(1000):
    batch_loss=[]
    for start in range(0,len(input_features),batch_size):
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
        prediction = model(xx)
        loss = cost(prediction, yy)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        batch_loss.append(loss.data.numpy())

    if i%100==0:
        losses.append(np.mean(batch_loss))
        print(f"第{i}次迭代，损失为{np.mean(batch_loss)}")

# 预测训练结果
x=torch.tensor(input_features,dtype=torch.float)
predict=model(x).data.numpy()

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})

months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]

test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predict.reshape(-1)})

# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')

# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = 60)
plt.legend()

# 图名
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values')
plt.show()