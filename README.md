# **PlantClassificationV3**
机器学习模型，用于分类从全消声室环境中采集到的植物发声信号。

### 项目简介  
PlantClassficationV3 是一个基于机器学习的项目，旨在对从全消声室环境中采集的植物发声信号进行分类。通过使用先进的音频处理技术和机器学习模型，项目能够帮助研究植物在不同环境压力下的声学反应，探索其与植物健康和生长状况的关系。

下图展示了本实验组通过摄像机捕捉到的植物发声瞬间：
<div align="center">
  <img src="https://github.com/user-attachments/assets/84199ad1-0e00-44d6-b68d-7b1b9e0aebf9" alt="植物发声图像" width="600"/>
</div>

### 依赖库及安装方法  
在使用该项目之前，请确保运行的环境中已安装以下库：  
- **numpy**：用于高效的数值计算
- **librosa**：音频处理库
- **torch** 和 **torchvision**：用于构建和训练深度学习模型
- **scikit-learn**：机器学习工具包，用于数据预处理和模型评估
- **matplotlib**：数据可视化库
- **PyWavelets**：用于信号处理的离散小波变换库
- **d2l**：用于深度学习教程的工具包

可以通过如下命令进行安装：  
```bash
pip install -r requirements.txt
```
推荐使用基于Pytorch的开发框架<https://pytorch.org/>
### 数据说明及准备         
植物发声数据以单通道wav文件格式存储，默认采样率为320000Hz，时长为4ms。将植物发声数据集放置在指定的目录下，每个类别的数据分别存放在不同的文件夹中，文件夹名称即为标签。确保数据的组织方式如下：  
```bash
/dataset/
    /class1/
        audio_file1.wav
        audio_file2.wav
    /class2/
        audio_file3.wav
        audio_file4.wav
```
请根据实际数据集进行适当调整。

### 使用说明
1.将项目克隆到本地：
```bash
git clone https://github.com/CrayonSh1n-chan/PlantClassificationV3.git
```
2.安装依赖：
```bash
pip install -r requirements.txt
```
3.准备数据，并按上面“数据准备”部分的要求将数据集放在指定目录中。

4.如果需要自定义训练参数，可以在代码中中调整相关设置，如学习率、批次大小等。

### 常见问题：
- **Python版本问题**：确保Python版本与依赖库兼容。可以通过 `python --version` 检查Python版本。
- **依赖安装问题**：如果遇到依赖安装错误，请确保所有库已正确安装，或者尝试使用虚拟环境进行隔离安装。
- **CUDA相关问题**：为了提升代码运行效率，请尽可能配置基于GPU版本的CUDA相关环境<https://developer.nvidia.com/accelerated-computing-toolkit>。
