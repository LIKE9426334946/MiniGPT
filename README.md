# MiniGPT
从零实现一个非常简单的大模型，构造了100个token词典，每个token是一个64维的向量

# 模型设计
模型设计参数都写在config.yaml文件中，方便以后修改  
共有2个Transformer block  
共有4个head  
feed forward维度为192  
使用ReLU激活函数  
使用CrossEntropyLoss计算损失函数  