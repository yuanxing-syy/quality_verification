# 眼底彩照粗筛及左右眼判断 
## 项目背景   
原始的眼底彩照数据中，存在大量拍摄质量低的图像，在使用之前，需要先对数据进行粗筛即保留拍摄质量好的，去除拍摄质量低的。  
<img src="https://github.com/yuanxing-syy/quality_verification/raw/master/src_images/数据展示.png" width="70%" height="70%"></img>  
## 实验要求
训练质量分类器，将标注数据中质量好的和质量差的分开
## 实现原理 
利用深度学习的方法，在pytorch框架上使用**resnet50**预训练模型实现医学图像的二分类
## 实验功能
能够对眼底彩照进行粗筛，预测眼底彩照拍摄质量好坏(默认score>0.5 属于符合要求的，score 越高，质量越好)； 并判断该图片是来自左眼还是右眼 
## 实验细节
<img src="https://github.com/yuanxing-syy/quality_verification/raw/master/src_images/实验细节.png" width="80%" height="80%"></img>  
## 预测结果demo可视化
<img src="https://github.com/yuanxing-syy/quality_verification/raw/master/src_images/预测结果demo可视化截图.png" width="70%" height="70%"></img>  
##### 1) Requirements  
virtualenv env  
source env/bin/activate  
pip install -r requirements.txt  
##### 2）主要依赖：  
python 2.x  
OpenCV 3.1.0 (import cv2能成功就行)  
numpy  1.12.1  
torch  1.0.0  
sklearn 0.20.0  
skimage 0.14.1  
##### 3）用法：  
先修改 test_config.json  
运行   python predictor.py  
输出   name_LR_score_final.txt  