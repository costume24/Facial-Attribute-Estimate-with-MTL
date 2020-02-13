这个版本主要是针对SE模块的实验。

实验的超参现在基本确定为：

lr: 1e-3  batchsize: 64 epoch: 30

每10个epoch降低一次学习率，第10个epoch加入loss的加权。（加权位置可能会随着后续对loss加权方式的改动而改变）

# 一、se模块加在主干道上

以下三个实验分别对应`psmcnn-se-1.py`，`psmcnn-se-2.py`，`psmcnn-se-1.py`。

## 1.1 只加tsnet

best acc：0.91935

模型文件不全，因此没有保存下来

多次实验的记录：

0.91935

0.9193

## 1.2 只加snet

best acc ： 0.9194

模型文件不全，因此没有保存下来

多次实验的记录：

0.9194

0.9181

## 1.3 tsnet和snet都加

best acc：0.9189

多次实验的记录：

0.9189