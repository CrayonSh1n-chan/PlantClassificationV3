import os
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy.signal as signal
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 参数设置
root_dir = r"C:\Users\nihao\Desktop\yc_dryN"  # 音频文件的根目录
sample_rate = 320000  # 采样率
duration = 0.018  # 音频片段长度，单位为秒
num_samples = int(sample_rate * duration)  # 根据采样率和持续时间计算的样本数
batch_size = 128  # 批次大小
num_epochs = 256  # 训练轮数
lr = 0.001  # 学习率

# 数据集类：用于处理音频数据的自定义Dataset类
class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # 音频文件的存储目录
        self.transform = transform  # 预处理函数
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]  # 获取目录下的所有类别
        self.file_paths = []  # 存储文件路径
        self.labels = []  # 存储标签

        # 遍历目录收集文件路径和对应的标签
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)  # 类别文件夹路径
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.endswith('.wav'):
                        self.file_paths.append(os.path.join(root, file))  # 添加音频文件路径
                        self.labels.append(idx)  # 添加对应的标签

    def __len__(self):
        return len(self.file_paths)  # 返回数据集中样本的总数

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]  # 获取文件路径
        label = self.labels[idx]  # 获取对应的标签
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)  # 加载音频文件

        # 如果音频长度小于预设长度，进行填充；如果过长，进行截断
        if len(audio) < num_samples:
            audio = np.pad(audio, (0, num_samples - len(audio)), 'constant')  # 用0填充
        else:
            audio = audio[:num_samples]  # 截断音频

        if self.transform:
            audio = self.transform(audio)  # 进行数据预处理
        return audio, label  # 返回音频数据和标签

# 数据预处理
transform = lambda x: torch.tensor(x, dtype=torch.float32)  # 将音频数据转换为张量

# 加载数据
dataset = AudioDataset(root_dir, transform=transform)  # 实例化AudioDataset类

# 划分训练集和验证集
train_size = int(0.75 * len(dataset))  # 训练集占75%
valid_size = len(dataset) - train_size  # 验证集占剩余的25%
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])  # 随机划分数据集

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 加载训练集
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)  # 加载验证集

# CNN 模型定义：用于音频分类的卷积神经网络
class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        # 定义多个卷积层和批归一化层
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)  # 第一个卷积层
        self.bn1 = nn.BatchNorm1d(64)  # 批归一化
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)  # 第二个卷积层
        self.bn2 = nn.BatchNorm1d(128)  # 批归一化
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)  # 第三个卷积层
        self.bn3 = nn.BatchNorm1d(256)  # 批归一化
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)  # 第四个卷积层
        self.bn4 = nn.BatchNorm1d(512)  # 批归一化
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 最大池化层

        self._to_linear = None
        # 定义卷积神经网络
        self.convs = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),  # ReLU激活函数
            self.pool,  # 最大池化层
            self.conv2,
            self.bn2,
            nn.ReLU(),
            self.pool,
            self.conv3,
            self.bn3,
            nn.ReLU(),
            self.pool,
            self.conv4,
            self.bn4,
            nn.ReLU(),
            self.pool
        )
        self.fc1 = nn.Linear(self._get_conv_output(num_samples), 1024)  # 第一个全连接层
        self.fc2 = nn.Linear(1024, 512)  # 第二个全连接层
        self.fc3 = nn.Linear(512, num_classes)  # 输出层

    def _get_conv_output(self, shape):
        # 通过一个虚拟数据来确定全连接层的输入尺寸
        o = self.convs(torch.zeros(1, 1, shape))
        self._to_linear = o.shape[1] * o.shape[2]  # 计算卷积层输出展平后的尺寸
        return self._to_linear

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度以匹配卷积层的要求
        x = self.convs(x)  # 卷积操作
        x = x.view(x.size(0), -1)  # 将多维输出展平为一维
        x = F.relu(self.fc1(x))  # 全连接层1并使用ReLU激活函数
        x = F.relu(self.fc2(x))  # 全连接层2并使用ReLU激活函数
        x = self.fc3(x)  # 输出分类结果
        return x

# 训练和验证函数：定义模型的训练和验证过程
def train_and_validate(net, train_loader, valid_loader, num_epochs, lr, device):
    net = net.to(device)  # 将模型加载到指定设备（CPU或GPU）
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 定义优化器，使用Adam优化算法
    loss = nn.CrossEntropyLoss()  # 定义损失函数，使用交叉熵损失

    for epoch in range(num_epochs):
        net.train()  # 设置模型为训练模式
        total_loss = 0  # 累计损失
        correct_train = 0  # 正确的训练预测数
        total_train = 0  # 总训练样本数

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)  # 将数据加载到指定设备
            optimizer.zero_grad()  # 清空梯度
            outputs = net(features)  # 前向传播，计算输出
            l = loss(outputs, labels)  # 计算损失
            l.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数
            total_loss += l.item()  # 累加损失

            # 计算训练准确率
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            correct_train += (predicted == labels).sum().item()  # 累加正确预测数
            total_train += labels.size(0)  # 累加总样本数

        train_acc = correct_train / total_train  # 计算训练准确率

        # 验证阶段
        net.eval()  # 设置模型为验证模式
        valid_loss = 0  # 验证损失
        correct_valid = 0  # 正确的验证预测数
        total_valid = 0  # 总验证样本数

        with torch.no_grad():
            for features, labels in valid_loader:
                features, labels = features.to(device), labels.to(device)  # 将数据加载到指定设备
                outputs = net(features)  # 前向传播，计算输出
                l = loss(outputs, labels)  # 计算损失
                valid_loss += l.item()  # 累加验证损失

                # 计算验证准确率
                _, predicted = torch.max(outputs, 1)  # 获取预测结果
                correct_valid += (predicted == labels).sum().item()  # 累加正确预测数
                total_valid += labels.size(0)  # 累加总样本数

        valid_acc = correct_valid / total_valid  # 计算验证准确率

        # 输出每轮的损失和准确率
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss / len(valid_loader):.4f}, '
              f'Valid Acc: {valid_acc:.4f}')

# 实例化模型并训练
num_classes = 4  # 定义类别数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备，优先使用GPU
net = AudioClassifier(num_classes)  # 实例化模型
train_and_validate(net, train_dataloader, valid_dataloader, num_epochs, lr, device)  # 开始训练和验证模型

# 生成并显示混淆矩阵
def evaluate_accuracy(data_iter, net, device):
    net.eval()  # 设置模型为验证模式
    metric = [0.0, 0.0]  # 用于记录准确率的列表
    all_preds = []  # 存储所有预测结果
    all_labels = []  # 存储所有真实标签
    with torch.no_grad():
        for features, labels in data_iter:
            features, labels = features.to(device), labels.to(device)  # 将数据加载到指定设备
            outputs = net(features)  # 前向传播，计算输出
            preds = outputs.argmax(axis=1)  # 获取预测结果
            metric[0] += (preds == labels).sum().item()  # 累加正确预测数
            metric[1] += labels.size(0)  # 累加总样本数
            all_preds.extend(preds.cpu().numpy())  # 收集预测结果
            all_labels.extend(labels.cpu().numpy())  # 收集真实标签
    return metric[0] / metric[1], all_preds, all_labels  # 返回准确率、预测结果和真实标签

# 评估训练集和验证集的准确率，并生成混淆矩阵
train_acc, train_preds, train_labels = evaluate_accuracy(train_dataloader, net, device)
valid_acc, valid_preds, valid_labels = evaluate_accuracy(valid_dataloader, net, device)

print(f'Train Accuracy: {train_acc:.4f}, Valid Accuracy: {valid_acc:.4f}')  # 输出训练集和验证集的准确率

# 混淆矩阵
cm = confusion_matrix(valid_labels, valid_preds)  # 计算混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(num_classes))  # 显示混淆矩阵

# 显示混淆矩阵
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax)  # 绘制混淆矩阵
plt.title('Confusion Matrix')  # 设置标题
plt.show()  # 显示图像
