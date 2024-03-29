# 保险回访外呼机器人

> 这是一个用于保险回访的外呼机器人，本项目中只保留核心的文本机器人的部分。
> 核心功能流程如下：


![](https://github.com/Ulov888/ReturnVisitRobot/blob/main/process%20(1).png)


本项目基于rasa重写了基于有限状态机的对话响应策略，对想要搭建文本机器人的朋友，应该有一定参考意义。

## 使用指南
配置，可修改confg.yml文件中的pipeline，pipeline的具体配置方法可以参考Rasa官网https://rasa.com/docs/rasa/model-configuration

下面给出两个自定义组件的配置示例

基于记忆性对话响应策略
```
policies:
  - name: policy.memoization.MemoizationPolicy
  - name: TEDPolicy
    max_history: 20
    epochs: 15
    batch_size: 50
  - name: RulePolicy
    core_fallback_threshold: 0.3
    enable_fallback_prediction: True
    core_fallback_action_name: "action_default_fallback"
```
或者基于有限状态机（FSM）
```
policies:
  - name: policy.fsm_policy.FsmPolicy
  - name: TEDPolicy
    max_history: 20
    epochs: 15
    batch_size: 50
  - name: RulePolicy
    core_fallback_threshold: 0.3
    enable_fallback_prediction: True
    core_fallback_action_name: "action_default_fallback"
```

训练
```
python main.py train
```
运行
```
#开启动作响应服务器，默认5055端口
python main.py run actions
#开启对话shell,如果使用pycharm注意勾选emulate terminal
python main.py shell
#第一句请输入内置意图：开始
Your input:开始
```
### Prerequisites 项目环境依赖


```
six~=1.16.0
rasa~=3.1.0
zlib~=1.2.12
tqdm~=4.64.0
```

### 安装
```
pip install -r requirements.txt
```


## 部署方法，可通过rest 方式调用会话响应
```
rasa run -m xx.tar.gz(模型文件) -p 8787(端口号)
```
