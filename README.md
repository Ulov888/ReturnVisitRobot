# Project Title 真实场景下的保险回访机器人

> 这是一个用于保险回访的外呼机器人，本项目中只保留核心的文本机器人的部分。

本项目基于rasa重写了基于有限状态机的对话响应策略，对想要搭建文本机器人的朋友，应该有一定参考意义。

![](https://github.com/dbader/readme-template/raw/master/header.png)

## Getting Started 使用指南

项目使用条件、如何安装部署、怎样运行使用以及使用演示

### Prerequisites 项目环境依赖


```
six~=1.16.0
rasa~=3.1.0
zlib~=1.2.12
tqdm~=4.64.0
```

### Installation 安装
'''
pip install -r requirement
'''
### Usage example 使用示例

给出更多使用演示和截图，并贴出相应代码。

## Deployment 部署方法，可通过rest 方式调用会话响应
'''
rasa run -m xx.tar.gz(模型文件) -p 8787(端口号)
'''
