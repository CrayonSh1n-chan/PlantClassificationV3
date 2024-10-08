PlantClassficationV3   

项目简介 	       
PlantClassficationV3 是一个基于机器学习的项目，旨在对从全消声室环境中采集的植物发声信号进行分类。该项目使用先进的音频处理技术和机器学习模型，帮助探索植物在胁迫环境中的发声。

依赖库及安装方法  
在使用该项目之前，请确保运行的环境中已安装以下库：      
numpy librosa torch torchvision scikit-learn matplotlib PyWavelets d2l      
可以通过如下命令进行安装：    
pip install -r requirements.txt

数据准备     
植物发声数据集放置在指定的目录下，其中每一相同类别数据放在同一文件夹中，文件夹名称即为标签

功能说明       
代码中包含有模型构建、数据预处理、特征工程、模型训练、模型评估及过程可视化
