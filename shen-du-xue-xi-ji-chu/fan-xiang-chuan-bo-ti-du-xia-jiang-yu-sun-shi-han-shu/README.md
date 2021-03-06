# 反向传播、梯度下降与损失函数

在深度学习中有以下三个概念不是很容易理解，他们分别为反向传播，梯度下降与损失函数。而且这三个概念之间关联非常紧密，我们接下来就依次对这三个概念进行分析。

神经网络整个训练过程我们可以总结如下：

* 猜一个结果a
* 看猜测的结果a和真实值y之间的差距
* 根据上面得到的差距调整策略，再猜一次
* 重复上面的操作，直到预测值和真实值之间已经的差距趋紧于0就停止训练

在神经网络训练中，我们把“猜”叫做初始化，可以随机，也可以根据以前的经验给定初始值。

ai-edu中有几个比较形象的例子，大家可以看一下：

{% embed url="https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/Step1%20-%20BasicKnowledge/02.0-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E4%B8%8E%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md" caption="反向传播与梯度下降" %}

## 总结

简单总结一下反向传播与梯度下降的基本工作原理：

1. 初始化；
2. 正向计算；
3. 损失函数为我们提供了计算损失的方法；
4. 梯度下降是在损失函数基础上向着损失最小的点靠近而指引了网络权重调整的方向；
5. 反向传播把损失值反向传给神经网络的每一层，让每一层都根据损失值反向调整权重；
6. Go to 2，直到精度足够好（比如损失函数值小于 0.001）。

