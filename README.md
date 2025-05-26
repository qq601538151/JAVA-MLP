# JAVA-MLP
测试类主要用公开数据集——鸳花数据集进行三分类测试，最终调整参数后，结果如下
# 不同的激活函数，损失函数结果

```
SIGMOID
```

Epoch 0, Loss: 0.322701
Epoch 100, Loss: 0.250213
Epoch 200, Loss: 0.250087
Epoch 300, Loss: 0.250110
Epoch 400, Loss: 0.250142
Epoch 500, Loss: 0.250087
Epoch 600, Loss: 0.250119
Epoch 700, Loss: 0.250112
Epoch 800, Loss: 0.250082
Epoch 900, Loss: 0.250092
Confusion Matrix:
[2, 0]
[2, 0]

```
RELU
```

Epoch 0, Loss: 0.563492
Epoch 100, Loss: 0.376186
Epoch 200, Loss: 0.375067
Epoch 300, Loss: 0.375098
Epoch 400, Loss: 0.375093
Epoch 500, Loss: 0.375060
Epoch 600, Loss: 0.375105
Epoch 700, Loss: 0.375100
Epoch 800, Loss: 0.375041
Epoch 900, Loss: 0.375116
Confusion Matrix:
[2, 0]
[2, 0]

```
TANH
```

Epoch 0, Loss: 0.741013
Epoch 100, Loss: 0.275517
Epoch 200, Loss: 0.267453
Epoch 300, Loss: 0.263540
Epoch 400, Loss: 0.259527
Epoch 500, Loss: 0.257545
Epoch 600, Loss: 0.257382
Epoch 700, Loss: 0.257304
Epoch 800, Loss: 0.257293
Epoch 900, Loss: 0.257272
Confusion Matrix:
[2, 0]
[1, 1]

# 代码调整，跑测试数据集

数据集采用公开数据集——鸢尾花数据集 iris.csv

下载的数据集一共三分类，最后结果也是三分类，通过调整学习率等参数，取得一个较好的结果
# 调整参数后的结果

```java
Epoch 0, Loss: 0.461787
Epoch 100, Loss: 0.066981
Epoch 200, Loss: 0.055269
Epoch 300, Loss: 0.052651
Epoch 400, Loss: 0.049791
Epoch 500, Loss: 0.047154
Epoch 600, Loss: 0.045214
Epoch 700, Loss: 0.043742
Epoch 800, Loss: 0.042521
Epoch 900, Loss: 0.041487
Epoch 1000, Loss: 0.040595
Epoch 1100, Loss: 0.039819
Epoch 1200, Loss: 0.039141
Epoch 1300, Loss: 0.038543
Epoch 1400, Loss: 0.038010
Epoch 1500, Loss: 0.037530
Epoch 1600, Loss: 0.037096
Epoch 1700, Loss: 0.036697
Epoch 1800, Loss: 0.036334
Epoch 1900, Loss: 0.035998
Epoch 2000, Loss: 0.035683
Epoch 2100, Loss: 0.035391
Epoch 2200, Loss: 0.035117
Epoch 2300, Loss: 0.034861
Epoch 2400, Loss: 0.034621
Epoch 2500, Loss: 0.034394
Epoch 2600, Loss: 0.034183
Epoch 2700, Loss: 0.033983
Epoch 2800, Loss: 0.033798
Epoch 2900, Loss: 0.033624
Confusion Matrix:
[50, 0, 0]
[0, 48, 2]
[0, 0, 50]
```

![image](https://github.com/user-attachments/assets/d092b711-c90d-4de4-8bb1-2703f2543cfd)
