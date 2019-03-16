# Tiger-face-detector-ssd

![Output1](https://github.com/Migasong/Tiger-face-detector-ssd/blob/master/config/output/6.jpg?raw=true)
![Output2](https://github.com/Migasong/Tiger-face-detector-ssd/blob/master/config/output/12.jpg?raw=true)



**环境：**
* Python 3.6
* Pytorch-GPU version
* OpenCV3

**文件夹说明：**
* 'logdir/'文件夹用于存储可供tensorboard使用的log文件
* 'model/'文件夹用于存储训练好的网络参数
* 'config/'文件夹中为程序的源码

**config文件夹说明：**
* 'dataset/'文件夹中为BBox-Label-Tool-master中整理好的’lb.txt‘文件，以及将其转换成的pkl文件'tiger.pkl'
* 'input/'文件夹用于存放测试集图片
* ‘output/’文件夹用于存放虎脸检测后的测试集图片
* ‘tiger_dataset/’文件夹用于存放训练集图片
* 'weights/'文件夹用于存放预训练的vgg16网络参数
* ‘settings.py’程序中存放了在其他程序中需要用到的各种路径和参数等，若要运行程序，首先需要自行修改settings文件中的路径
* ‘ssd.py’为主要的model代码，其中使用到了‘l2norm.py’文件中的函数
* ‘dataset.py’里包含了图像预处理，准备训练集和测试集的代码
* 'train.py'为训练代码
* ‘test.py’为测试代码
* 'multibox_loss.py'为训练中要用到的loss函数
* ‘box_utils.py’中存放了如nms，jaccard，match等重要的函数供其他程序使用
* ‘read_txt.py’为将‘lb.txt’转换为‘tiger.pkl’以供dataset使用的代码
* ‘prior_box.py’为准备default boxes的代码
* 'config.py'中存放了一些其他程序会使用到的参数

**使用说明：**
* 修改’settings.py‘中的各文件路径
* 运行’train.py‘程序可对数据进行训练
* 运行’test.py‘程序可对数据进行测试
* 该程序为GPU版本，若想在CPU上使用需要修改‘train.py’中的下述代码
(1) 注释line 31, line 140
``` python
 31	torch.cuda.set_device(1)
```
``` python
140	data = data.cuda()
```
(2) 注释line 57, 使用line 58
``` python
 57	self.net = self.net.cuda()
 58	#self.net = self.net
```
(3) MultiBoxLoss最后一个参数表示是否使用GPU，将其由True改为False
``` python
121	crit = MultiBoxLoss(voc['num_classes'], 0.5, True, 0, True, 3, 0.5, False, True)
```
(4) 此外，还有一些数据格式的变化需自行根据报错修改
