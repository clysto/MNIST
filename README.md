# 全连接网络识别手写数字

使用两层全连接层实现神经网络，在MNIST数据集上训练。

## 开发

```bash
# 创建conda虚拟环境
conda env create -f environment.yml

# 激活虚拟环境
conda activate fastapi
```

## 使用

> **在使用cli测试或者开启服务器之前先运行`python -m main train`，保证已经拥有训练模型后再启动进行测试。**

```bash
# 训练
# 训练会将模型参数存于data/model.pl中
python -m main train

# 将测试集中的前100张图片解压到data/extract目录中
python -m main extract

# 预测IMAGE PATH文件中的手写数字
python -m cli <IMAGE PATH>

# 开启web服务器
uvicorn app.main:app
```

## web

![web](./data/web.png)
