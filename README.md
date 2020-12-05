# 全连接网络识别手写数字

使用两层全连接层实现神经网络，在MNIST数据集上训练。

## 使用


```bash
# 训练
# 训练会将模型参数存于data/model.pl中
python -m main train

# 将测试集中的前100张图片解压到data/extract目录中
python -m main extract

# 预测IMAGE PATH文件中的手写数字
python -m test <IMAGE PATH>

# 开启web服务器
uvicorn app.main:app
```

## web

![web](./data/web.png)
