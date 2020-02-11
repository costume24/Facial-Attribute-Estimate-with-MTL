- 实验内容：ADAM优化器+新的初始化
- 超参：

  - lr:1-e3

  - epoch:20
  - weight-decay:1e-4
- 其他：加入了师弟的loss加权
- 结果：不太好，收敛较慢



- 实验内容：ADAM+默认初始化
- 超参：
  - lr:1e-3
  - epoch:20
  - weight-decay:1e-4
- 其他：加入了师弟的loss加权
- 结果：acc 0.9184，loss 0.2338