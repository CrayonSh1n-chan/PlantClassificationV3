import os
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 参数设置
root_dir = r"C:\Users\nihao\Desktop\yc_dryN"  # 音频数据的根目录
sample_rate = 320000  # 音频采样率
duration = 0.018  # 每个音频片段的持续时间（上限）
num_samples = int(sample_rate * duration)  # 每个音频片段的采样点数
n_mfcc = 50  # MFCC特征数量
n_fft = 1024  # FFT窗口大小
batch_size = 256  # 每个批次的样本数
num_epochs = 320  # 训练的轮次
lr = 0.001  # 学习率
best_path = r"C:\Users\nihao\Desktop\model_path\yc4mfcc"  # 保存最佳模型的文件夹路径

# 如果文件夹不存在，创建它
os.makedirs(best_path, exist_ok=True)


# 特征提取函数
def extract_features(audio, sample_rate):
    features = []
    # 长度判断
    if len(audio) < num_samples:
        audio = np.pad(audio, (0, num_samples - len(audio)), 'constant')
    else:
        audio = audio[:num_samples]

    # 计算能量特征
    frame_length = int(0.010 * sample_rate)  # 帧长度为10毫秒
    hop_length = int(0.0025 * sample_rate)  # 帧移为2.5毫秒
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    features.append(energy)

    # 计算过零率特征
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    features.append(zcr)

    # 计算MFCC特征
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    features.append(mfccs.flatten())

    # 将所有特征连接成一个向量
    return np.concatenate(features)


# 数据集类
class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.file_paths = []
        self.labels = []

        # 遍历所有类别文件夹，收集音频文件路径和对应的标签
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.endswith('.wav'):
                        self.file_paths.append(os.path.join(root, file))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.file_paths)  # 返回数据集的大小

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]  # 获取音频文件路径
        label = self.labels[idx]  # 获取对应的标签
        audio, sr = librosa.load(file_path, sr=None, mono=True)  # 加载音频
        features = extract_features(audio, sample_rate)  # 提取特征
        if self.transform:
            features = self.transform(features)  # 应用变换
        return features, label  # 返回特征和标签


# 数据预处理
transform = lambda x: torch.tensor(x, dtype=torch.float32)  # 将特征转换为Tensor

# 加载数据
dataset = AudioDataset(root_dir, transform=transform)
train_size = int(0.8 * len(dataset))  # 80%的数据用于训练
valid_size = len(dataset) - train_size  # 剩下的20%用于验证
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])  # 划分数据集
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练集的数据加载器
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)  # 验证集的数据加载器

# 计算特征的维度
dummy_audio = np.zeros(num_samples)
dummy_features = extract_features(dummy_audio, sample_rate)
input_dim = dummy_features.shape[0]  # 输入的特征维度


# CNN 模型定义
class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)  # 第一个卷积层
        self.bn1 = nn.BatchNorm1d(64)  # 第一个批量归一化层
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)  # 第二个卷积层
        self.bn2 = nn.BatchNorm1d(128)  # 第二个批量归一化层
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)  # 第三个卷积层
        self.bn3 = nn.BatchNorm1d(256)  # 第三个批量归一化层
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)  # 第四个卷积层
        self.bn4 = nn.BatchNorm1d(512)  # 第四个批量归一化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 最大池化层

        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),  # 激活函数
            self.pool,
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

        # 计算全连接层的输入尺寸
        self._to_linear = self.calculate_linear_size()
        self.fc1 = nn.Linear(self._to_linear, 128)  # 全连接层1
        self.fc2 = nn.Linear(128, num_classes)  # 全连接层2（输出层）

    def calculate_linear_size(self):
        x = torch.randn(1, 1, input_dim)  # 创建一个假数据来计算尺寸
        x = self.convs(x)
        return x.shape[1] * x.shape[2]

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加一个维度以适应Conv1d
        x = self.convs(x)
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 模型实例化
num_classes = len(dataset.classes)  # 类别数量
model = AudioClassifier(num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam优化器

# 训练和验证
best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for features, labels in train_dataloader:
        optimizer.zero_grad()  # 梯度清零
        outputs = model(features)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数

        running_loss += loss.item()

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in valid_dataloader:
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # 计算准确率
    acc = accuracy_score(all_labels, all_preds)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(best_path, 'best_model.pth'))  # 保存最佳模型

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataloader)}, Accuracy: {acc}')

# 加载最佳模型
model.load_state_dict(torch.load(os.path.join(best_path, 'best_model.pth')))

# 混淆矩阵计算
model.eval()  # 设置模型为评估模式
all_preds = []  # 存储所有的预测结果
all_labels = []  # 存储所有的真实标签

with torch.no_grad():  # 在评估过程中不需要计算梯度，节省内存
    for features, labels in valid_dataloader:  # 遍历验证集
        features = features.unsqueeze(1).to(device)  # 添加通道维度，并移动到GPU
        labels = labels.to(device)  # 将标签移动到GPU
        outputs = model(features)  # 通过模型计算输出
        _, preds = torch.max(outputs, 1)  # 获取每个样本的预测类别
        all_preds.extend(preds.cpu().numpy())  # 把预测结果添加到列表中
        all_labels.extend(labels.cpu().numpy())  # 将真实标签添加到列表中

# 计算混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_preds)  # 计算混淆矩阵

# 显示混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=dataset.classes)
disp.plot(cmap=plt.cm.Blues)  # 使用蓝色配色显示混淆矩阵
plt.title("Confusion Matrix")  # 设置图表标题
plt.show()  # 显示图表

# 计算准确率
accuracy = accuracy_score(all_labels, all_preds)  # 计算准确率
print(f"Validation Accuracy: {accuracy * 100:.2f}%")  # 打印准确率

