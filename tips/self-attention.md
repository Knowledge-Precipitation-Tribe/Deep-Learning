# self-attention

> 转载自：[https://blog.csdn.net/qq\_40027052/article/details/78421155](https://blog.csdn.net/qq_40027052/article/details/78421155) by 张俊林

## Self Attention

Self Attention也经常被称为intra Attention（内部Attention），最近一年也获得了比较广泛的使用，比如Google最新的机器翻译模型内部大量采用了Self Attention模型。在一般任务的Encoder-Decoder框架中，输入Source和输出Target内容是不一样的，比如对于英-中机器翻译来说，Source是英文句子，Target是对应的翻译出的中文句子，Attention机制发生在Target的元素Query和Source中的所有元素之间。而Self Attention顾名思义，指的不是Target和Source之间的Attention机制，而是Source内部元素之间或者Target内部元素之间发生的Attention机制，也可以理解为Target=Source这种特殊情况下的注意力计算机制。其具体计算过程是一样的，只是计算对象发生了变化而已，所以此处不再赘述其计算过程细节。

如果是常规的Target不等于Source情形下的注意力计算，其物理含义正如上一节所讲，比如对于机器翻译来说，本质上是目标语单词和源语单词之间的一种单词对齐机制。

那么如果是Self Attention机制，一个很自然的问题是：通过Self Attention到底学到了哪些规律或者抽取出了哪些特征呢？或者说引入Self Attention有什么增益或者好处呢？

我们仍然以机器翻译中的Self Attention来说明，下面两个图可视化地表示Self Attention在同一个英语句子内单词间产生的联系。

![&#x56FE;1](../.gitbook/assets/image%20%2817%29.png)

![&#x56FE;2](../.gitbook/assets/image%20%2816%29.png)

从上面两张图可以看出，Self Attention可以捕获同一个句子中单词之间的一些句法特征（比如图1展示的有一定距离的短语结构）或者语义特征（比如图2展示的its的指代对象Law）。

很明显，引入Self Attention后会更容易捕获句子中长距离的相互依赖的特征，因为如果是RNN或者LSTM，需要依次序序列计算，对于远距离的相互依赖的特征，要经过若干时间步步骤的信息累积才能将两者联系起来，而距离越远，有效捕获的可能性越小。但是Self Attention在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来，所以远距离依赖特征之间的距离被极大缩短，有利于有效地利用这些特征。除此外，Self Attention对于增加计算的并行性也有直接帮助作用。这是为何Self Attention逐渐被广泛使用的主要原因。

