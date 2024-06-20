---
title: 基于Pytorch框架的深度学习入门基础
date: 2023/3/14 21:06:41
description: 深度学习入门，让你的机器产生思维吧！

categories: Deeplearn
tags: 深度学习
cover: https://pic1.zhimg.com/v2-5ea151d43c0ebb13546985d225ac256a_1200x500.jpg
---



## 首先声明！！！

---

* 1.脚本为本人总结，如有使用注明出处。
* 2.模型基于Pytorch框架实现及训练。
* 3.脚本内有注释。

---



## 一、main_demo

```python
# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
```



## 二、read_data（读取数据）

```python
from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):

    # 初始化函数
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    # 获取数据集
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        lable = self.label_dir
        return img, lable

    # 获取长度
    def __len__(self):
        return len(self.img_path)


root_dir = "dataset/hymenoptera_data/train"

ants_label_dir = "ants"
bees_label_dir = "bees"

ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset

```



## 三、test_tb（简单测试）

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image


writer = SummaryWriter("logs")
image_path = "dataset/hymenoptera_data/train/bees/16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
# print(type(img_array))
# print(img_array.shape)


writer.add_image("test", img_array, 2, dataformats='HWC')
# y = x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)


writer.close()
```

### 可视化页面展示：

![test1](test1.png)



## 四、Transforms（数据转换）

```python
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


# python的用法 -> tensor数据类型
# 通过 transforms.ToTensor看两个问题
# 1. transforms该如何使用？
# 2. 为什么我们需要Tensor数据类型

# 绝对路径 C:\Users\Administrator\Desktop\编程代码\Python\pytorch深度学习\dataset\hymenoptera_data\train\ants\0013035.jpg
# 相对路径 dataset/hymenoptera_data/train/ants/0013035.jpg

img_path = "dataset/hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("Tensor_img", tensor_img)

writer.close()
```



## 五、UseTransforms（使用数据转换）
```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


writer = SummaryWriter("logs")
img = Image.open("images/桌面.jpg")

# TpTensor的使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Totensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([6, 3, 2], [9, 3, 5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
```



## 六、dataset_transforms（数据集使用）

```python
import torchvision
from torch.utils.tensorboard import SummaryWriter


dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

# print(test_set[0])

writer = SummaryWriter("../p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image(("test_set", img, i))

writer.close()
```



## 七、read_data（读取数据）

```python
from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):

    # 初始化函数
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    # 获取数据集
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        lable = self.label_dir
        return img, lable

    # 获取长度
    def __len__(self):
        return len(self.img_path)


root_dir = "../dataset/hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset
```



## 八、nn_module（模型基础）

```python
import torch
from torch import nn


class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)
```



## 九、nn_conv2d（卷积层）

```python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("../data", train=False,
        transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()

writer = SummaryWriter("../logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    # print(imgs.shape)
    # print(output.shape)

    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)

    # torch.Size([64, 6, 30, 30]) -> [xxx, 3, 30, 30]
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step += 1
```



## 十、nn_maxpool（最大池化）

```python
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

# 二维矩阵最大池化
# input = torch.tensor([[1, 2, 0, 3, 1],
#                      [0, 1, 2, 3, 1],
#                      [1, 2, 1, 0, 0],
#                      [5, 2, 3, 1, 1],
#                      [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

tudui = Tudui()

writer = SummaryWriter("../logs_maxpool")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
```



## 十一、nn_relu（非线性激活）
```python
import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


input = torch.tensor([[1, -0.5],
                       [-1, 3]])


input = torch.reshape(input, (-1, 1, 2, 2))


dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


tudui = Tudui()
# output = tudui(input)
# print(output)


writer = SummaryWriter("../logs_relu")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
```



## 十二、nn_linear（线性层）

```python
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader


dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.reshape(imgs, (1, 1, 1, -1))
    print(output.shape)
    output = tudui(output)
    print(output.shape)
```



## 十三、nn_seq（搭建小实战）

```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):

    def __init__(self):
        super(Tudui, self).__init__()

        # self.conv1 = Conv2d(3, 32, 5, padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):

        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)

        x = self.model1(x)
        return x


tudui = Tudui()
print(tudui)

input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)


writer = SummaryWriter("../logs_seq")
writer.add_graph(tudui, input)

writer.close()
```



## 十四、损失函数与反向传播

### 1.nn_loss
```python
import torch
from torch.nn import L1Loss
from torch import nn


inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction='sum')
result = loss(inputs, targets)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)

print(result)
print(result_mse)
```
### 2.nn_loss_network
```python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 导入数据集
dataset = torchvision.datasets.CIFAR10("../data", train=False,
        transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)


# 神经网络模板
class Tudui(nn.Module):

    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# 计算损失
loss = nn.CrossEntropyLoss()
tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
    result_loss = loss(outputs, targets)
    result_loss.backward()
```

### 3.可视化页面展示

![test2](test2.png)



## 十五、nn_optim（优化器）

```python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 导入数据集
dataset = torchvision.datasets.CIFAR10("../data", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)


# 神经网络模板
class Tudui(nn.Module):

    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# 计算损失、梯度下降

loss = nn.CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss
    print(running_loss)
```



## 十六、model_pretrained（对现有网络模型进行修改.vgg16）

```python
import torchvision
from torch import nn


# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train',
#                                            download=True, transform=torchvision.transforms.ToTensor())


vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)



train_data = torchvision.datasets.CIFAR10("../data", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)


vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
```



## 十七、模型的保存与读取

### 1.model_save
```python
import torch
import torchvision


vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1:模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2:模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
```

### 2.model_load
```python
import torch
import torchvision


# 方式1：加载模型
# model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2：加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
print(vgg16)
```



## 十八、train（完整模型训练套路）

```python
import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *


# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../data", train=True,
                                           download=True, transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(root="../data", train=False,
                                           download=True, transform=torchvision.transforms.ToTensor())


# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10


# 添加tensorboard
writer = SummaryWriter("../logs_train")


# 开始训练
for i in range(epoch):

    print("-----第{}轮训练开始-----".format(i+1))

    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 显示次数和损失loss
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("teat_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # torch.save(tudui, "model_classify_{}.pth".format(i))
    # print("模型已保存！")


writer.close()
```
### 可视化页面展示：

![test3](test3.png)



## 十九、利用GPU训练

### 1.train_gpu_1
```python
import time
from model import *
import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../data", train=True,
                                           download=True, transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(root="../data", train=False,
                                           download=True, transform=torchvision.transforms.ToTensor())


# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()

if torch.cuda.is_available():
    tudui = tudui.cuda()


# 损失函数
loss_fn = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)


# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10


# 添加tensorboard
writer = SummaryWriter("../logs_train")
start_time = time.time()


# 开始训练
for i in range(epoch):

    print("-----第{}轮训练开始-----".format(i+1))

    # 训练步骤开始
    tudui.train()

    for data in train_dataloader:
        imgs, targets = data

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 显示次数和损失loss
        total_train_step += 1
        if total_train_step % 1000 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("teat_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    if i == 29:
        torch.save(tudui, "model_{}.pth".format(i+1))
        print("模型已保存！")


writer.close()
```

### 2.train_gpu_2
```python
import time
from model import *
import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../data", train=True,
                download=True, transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(root="../data", train=False,
                download=True, transform=torchvision.transforms.ToTensor())


# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()
tudui.to(device)


# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)


# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 30


# 添加tensorboard
writer = SummaryWriter("../logs_train")
start_time = time.time()


# 开始训练
for i in range(epoch):

    print("-----第{}轮训练开始-----".format(i+1))

    # 训练步骤开始
    tudui.train()

    for data in train_dataloader:
        imgs, targets = data

        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 显示次数和损失loss
        total_train_step += 1
        if total_train_step % 100 == 0:
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data

            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    end_time = time.time()
    print("time: {}s".format(end_time - start_time))
    print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("teat_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    if i == 29:
        torch.save(tudui, "model_{}.pth".format(i + 1))
        print("模型已保存！")


writer.close()
```



## 二十、test（模型验证）

```python
import torch
import torchvision
from torch import nn
from PIL import Image
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


image_path = "../images/dog.png"
image = Image.open(image_path)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)


class Tudui(nn.Module):

    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

model = torch.load("model_30.pth", map_location=torch.device('cpu'))

image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)

print(output.argmax(1))
```

---