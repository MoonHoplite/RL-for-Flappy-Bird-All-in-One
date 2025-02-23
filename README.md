# Flappy Bird 强化学习 All in One

## 介绍
这个项目实现了多种强化学习方法，并以 Flappy Bird 为例进行展示。

[![Watch the video](https://github.com/MoonHoplite/RL-for-Flappy-Bird-All-in-One/blob/master/assets/thumbnail.jpg)](https://github.com/user-attachments/assets/4d9a745b-2aec-42e4-94d8-845fb9ec05f0)

## 方法
* Value-based 方法: DQN (Deep Q-Network), DDQN (Dueling Deep Q-Network)
* Policy-based 方法: VPG (Vanilla Policy Gradient), PPO (Proximal Policy Gradient)

## 注意
由于采样的频率是每一帧, 随机初始化的情况下, 小鸟跳跃的概率是 50%, 会导致循环撞天花板，无法探索其他的路径. 因此对于 DQN 方法, 在观察阶段设置跳跃概率为 10%, 对于 policy-based 方法, 在策略网络初始化时手动设置最后一个线形层的 bias, 使其跳跃概率降低.   
对于 Flappy Bird 而言, Q-Learning 似乎比 Policy-based 方法效果好得多. DDQN 仅需训练 2-3 小时就可以达到100分以上, 但 Policy-based 方法收敛似乎很慢 (也可能是代码有bug). 如有有效的训练方法请指教.   
如果想自己玩, 可以启动 start_game.py, 但记得去 \game\wrapped_flappy_bird.py 手动调整帧率 ~~不然会怀疑人生~~

## 参考
本项目的完成离不开下面的项目.
* FlappyBird-PPO-pytorch https://github.com/luozhiyun993/FlappyBird-PPO-pytorch
* DeepLearningFlappyBird https://github.com/yenchenlin/DeepLearningFlappyBird
* FlapPyBird https://github.com/sourabhv/FlapPyBird
* spinningup https://github.com/openai/spinningup

