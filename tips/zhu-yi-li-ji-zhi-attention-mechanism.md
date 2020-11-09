# 注意力机制\(Attention Mechanism\)

> 转载自：[https://blog.csdn.net/qq\_40027052/article/details/78421155](https://blog.csdn.net/qq_40027052/article/details/78421155) by 张俊林

## 什么是注意力

人在观察事物时会有选择性的关注较为重要的信息，这个我们称其为注意力，然后通过持续关注这一关键位置以获得更多的信息，而忽略其他的无用信息。这种视觉上的注意力机制大大的提高了我们处理信息的效率与准确性。

![](../.gitbook/assets/image%20%2818%29.png)

上图形象化展示了人类在看到一副图像时是如何高效分配有限的注意力资源的，其中红色区域表明视觉系统更关注的目标，很明显对于上图所示的场景，人们会把注意力更多投入到人的脸部，文本的标题以及文章首句等位置。

## Attention模型

在上一小节中我们提到的Encoder-Decoder模型是没有注意力机制，所以可以把它称为注意力不集中的模型。为什么说它注意力不集中呢？请观察下目标句子Target中每个单词的生成过程如下：

$$
\begin{array}{l}
\mathbf{y}_{1}=\mathbf{f}(\mathbf{C}) \\
\mathbf{y}_{2}=\mathbf{f}\left(\mathbf{C}, \mathbf{y}_{1}\right) \\
\mathbf{y}_{3}=\mathbf{f}\left(\mathbf{C}, \mathbf{y}_{1}, \mathbf{y}_{2}\right)
\end{array}
$$

其中f是Decoder的非线性变换函数。从这里可以看出，在生成目标句子的单词时，不论生成哪个单词，它们使用的输入句子Source的语义编码C都是一样的，没有任何区别。而语义编码C是由句子Source的每个单词经过Encoder 编码产生的，这意味着不论是生成哪个单词，$$y_1$$，$$y_2$$还是$$y_3$$，其实句子Source中任意单词对生成某个目标单词$$y_i$$来说影响力都是相同的，这是为何说这个模型没有体现出注意力的缘由。这类似于人类看到眼前的画面，但是眼中却没有注意焦点一样。

如果拿机器翻译来解释这个分心模型的Encoder-Decoder框架更好理解，比如输入的是英文句子：Tom chase Jerry，Encoder-Decoder框架逐步生成中文单词：“汤姆”，“追逐”，“杰瑞”。在翻译“杰瑞”这个中文单词的时候，分心模型里面的每个英文单词对于翻译目标单词“杰瑞”贡献是相同的，很明显这里不太合理，显然“Jerry”对于翻译成“杰瑞”更重要，但是分心模型是无法体现这一点的，这就是为何说它没有引入注意力的原因。

没有引入注意力的模型在输入句子比较短的时候问题不大，但是如果输入句子比较长，此时所有语义完全通过一个中间语义向量来表示，单词自身的信息已经消失，可想而知会丢失很多细节信息，这也是为何要引入注意力模型的重要原因。

上面的例子中，如果引入Attention模型的话，应该在翻译“杰瑞”的时候，体现出英文单词对于翻译当前中文单词不同的影响程度，比如给出类似下面一个概率分布值：

$$
(\mathrm{Tom}, 0.3) (\mathrm{Chase}, 0.2) (\mathrm{Jerr} y, 0.5)
$$

每个英文单词的概率代表了翻译当前单词“**杰瑞**”时，注意力分配模型分配给不同英文单词的注意力大小。这对于正确翻译目标语单词肯定是有帮助的，因为引入了新的信息。同理，目标句子中的每个单词都应该学会其对应的源语句子中单词的注意力分配概率信息。

这意味着在生成每个单词$$y_i$$的时候，原先都是相同的中间语义表示C会被替换成根据当前生成单词而不断变化的$$C_i$$。理解Attention模型的关键就是这里，即由固定的中间语义表示C换成了根据当前输出单词来调整成加入注意力模型的变化的$$C_i$$。增加了注意力模型的Encoder-Decoder框架理解起来如下图所示。

![](../.gitbook/assets/image%20%2814%29.png)

即生成目标句子单词的过程成了下面的形式：

$$
\begin{array}{l}
\mathbf{y}_{\mathbf{1}}=\mathbf{f} \mathbf{1}\left(\mathbf{C}_{\mathbf{1}}\right) \\
\mathbf{y}_{\mathbf{2}}=\mathbf{f} \mathbf{1}\left(\mathbf{C}_{\mathbf{2}}, \mathbf{y}_{\mathbf{1}}\right) \\
\mathbf{y}_{\mathbf{3}}=\mathbf{f} \mathbf{1}\left(\mathbf{C}_{\mathbf{3}}, \mathbf{y}_{\mathbf{1}}, \mathbf{y}_{\mathbf{2}}\right)
\end{array}
$$

而每个$$C_i$$可能对应着不同的源语句子单词的注意力分配概率分布，比如对于上面的英汉翻译来说，其对应的信息可能如下：

![](https://cdn.mathpix.com/snip/images/dz_3AJ5NDITGa57GOORsWWGQU4-HeUwSRQpoyLZigO4.original.fullsize.png)

其中，f2函数代表Encoder对输入英文单词的某种变换函数，比如如果Encoder是用的RNN模型的话，这个f2函数的结果往往是某个时刻输入$$x_i$$后隐层节点的状态值；g代表Encoder根据单词的中间表示合成整个句子中间语义表示的变换函数，一般的做法中，g函数就是对构成元素加权求和，即下列公式:

$$
C_{i}=\sum_{j=1}^{L_{x}} a_{i j} h_{j}
$$

其中，$$L_x$$代表输入句子Source的长度，$$a_{ij}$$代表在Target输出第i个单词时Source输入句子中第$$j$$个单词的注意力分配系数，而$$h_j$$则是Source输入句子中第$$j$$个单词的语义编码。假设$$C_i$$下标$$i$$就是上面例子所说的“ 汤姆” ，那么$$L_x$$就是3，$$h1=f("Tom")，h2=f("Chase"),h3=f("Jerry")$$分别是输入句子每个单词的语义编码，对应的注意力模型权值则分别是0.6,0.2,0.2，所以g函数本质上就是个加权求和函数。如果形象表示的话，翻译中文单词“汤姆”的时候，数学公式对应的中间语义表示$$C_i$$的形成过程类似下图。

![](../.gitbook/assets/image%20%287%29.png)

这里还有一个问题：生成目标句子某个单词，比如“汤姆”的时候，如何知道Attention模型所需要的输入句子单词注意力分配概率分布值呢？就是说“汤姆”对应的输入句子Source中各个单词的概率分布：$$(Tom,0.6)(Chase,0.2) (Jerry,0.2)$$ 是如何得到的呢？

我们将上一小节中的的非Attention模型的Encoder-Decoder框架进行细化，Encoder采用RNN模型，Decoder也采用RNN模型，这是比较常见的一种模型配置，则Encoder-Decoder的框架转换为下图。

![](../.gitbook/assets/image%20%2811%29.png)

那么用下图可以较为便捷地说明注意力分配概率分布值的通用计算过程。

![](../.gitbook/assets/image%20%288%29.png)

对于采用RNN的Decoder来说，在时刻$$i$$，如果要生成$$y_i$$单词，我们是可以知道Target在生成$$y_i$$之前的时刻$$i-1$$时，隐层节点i-1时刻的输出值$$H_{i-1}$$的，而我们的目的是要计算生成$$y_i$$时输入句子中的单词“Tom”、“Chase”、“Jerry”对$$y_i$$来说的注意力分配概率分布，那么可以用Target输出句子$$i-1$$时刻的隐层节点状态$$H_{i-1}$$去一一和输入句子Source中每个单词对应的RNN隐层节点状态$$h_j$$进行对比，即通过函数$$F(h_j,H_{i-1})$$来获得目标单词$$y_i$$和每个输入单词对应的对齐可能性。

（待完善）

这个$$F$$函数在不同论文里可能会采取不同的方法，然后函数F的输出经过Softmax进行归一化就得到了符合概率分布取值区间的注意力分配概率分布数值。

绝大多数Attention模型都是采取上述的计算框架来计算注意力分配概率分布信息，区别只是在$$F$$的定义上可能有所不同。下图可视化地展示了在英语-德语翻译系统中加入Attention机制后，Source和Target两个句子每个单词对应的注意力分配概率分布。

![](../.gitbook/assets/image%20%2813%29.png)

上述内容就是经典的Soft Attention模型的基本思想，那么怎么理解Attention模型的物理含义呢？

一般在自然语言处理应用里会把Attention模型看作是输出Target句子中某个单词和输入Source句子每个单词的对齐模型，这是非常有道理的。目标句子生成的每个单词对应输入句子单词的概率分布可以理解为输入句子单词和这个目标生成单词的对齐概率，这在机器翻译语境下是非常直观的：传统的统计机器翻译一般在做的过程中会专门有一个短语对齐的步骤，而注意力模型其实起的是相同的作用。

## Attention的本质

如果把Attention机制从上文讲述例子中的Encoder-Decoder框架中剥离，并进一步做抽象，可以更容易看懂Attention机制的本质思想。

![](../.gitbook/assets/image%20%2815%29.png)

我们可以这样来看待Attention机制（参考上图）：将Source中的构成元素想象成是由一系列的&lt; Key,Value &gt;数据对构成，此时给定Target中的某个元素Query，通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。所以本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数。即可以将其本质思想改写为如下公式：

$$
\text { Attention(Query, Source) }=\sum_{i=1}^{L_{x}} \text { Similarity(Query, Key}_i )\text { * Value}_{i}
$$

其中， $$L_x=||Source||$$代表Source的长度，公式含义即如上所述。

上文所举的机器翻译的例子里，因为在计算Attention的过程中，Source中的Key和Value合二为一，指向的是同一个东西，也即输入句子中每个单词对应的语义编码，所以可能不容易看出这种能够体现本质思想的结构。

当然，从概念上理解，把Attention仍然理解为从大量信息中有选择地筛选出少量重要信息并聚焦到这些重要信息上，忽略大多不重要的信息，这种思路仍然成立。

聚焦的过程体现在权重系数的计算上，权重越大越聚焦于其对应的Value值上，即权重代表了信息的重要性，而Value是其对应的信息。

对于注意力机制可以引出另外一种理解，也可以将Attention机制看作一种软寻址（Soft Addressing）:Source可以看作存储器内存储的内容，元素由地址Key和值Value组成，当前有个Key=Query的查询，目的是取出存储器中对应的Value值，即Attention数值。通过Query和存储器内元素Key的地址进行相似性比较来寻址，之所以说是软寻址，指的不像一般寻址只从存储内容里面找出一条内容，而是可能从每个Key地址都会取出内容，取出内容的重要性根据Query和Key的相似性来决定，之后对Value进行加权求和，这样就可以取出最终的Value值，也即Attention值。所以不少研究人员将Attention机制看作软寻址的一种特例，这也是非常有道理的。

至于Attention机制的具体计算过程，如果对目前大多数方法进行抽象的话，可以将其归纳为两个过程：

* 第一个过程是根据Query和Key计算权重系数
  * 第一个阶段根据Query和Key计算两者的相似性或者相关性
  * 第二个阶段对第一阶段的原始分值进行归一化处理
* 第二个过程根据权重系数对Value进行加权求和

这样，可以将Attention的计算过程抽象为如下图展示的三个阶段。

![](../.gitbook/assets/image%20%289%29.png)

在第一个阶段，可以引入不同的函数和计算机制，根据Query和某个$$\text{Key}_i$$，计算两者的相似性或者相关性，最常见的方法包括：

* 求两者的向量点积
* 求两者的向量Cosine相似性
* 通过再引入额外的神经网络来求值

第一阶段产生的分值根据具体产生的方法不同其数值取值范围也不一样，第二阶段引入类似SoftMax的计算方式对第一阶段的得分进行数值转换，一方面可以进行归一化，将原始计算分值整理成所有元素权重之和为1的概率分布；另一方面也可以通过SoftMax的内在机制更加突出重要元素的权重。即一般采用如下公式计算：

$$
a_{i}=\text { Softmax} \left(\operatorname{Sim}_{i}\right)=\frac{e^{\operatorname{sim}_{i}}}{\sum_{j=1}^{L_{x}} e^{\operatorname{sim}_{j}}}
$$

第二阶段的计算结果$$a_i$$即为$$\text{Value}_i$$对应的权重系数，然后进行加权求和即可得到Attention数值：

$$
\text { Attention(Query, Source) }=\sum_{i=1}^{L_{x}} a_{i} \cdot \text { Value }_{i}
$$

通过如上三个阶段的计算，即可求出针对Query的Attention数值，目前绝大多数具体的注意力机制计算方法都符合上述的三阶段抽象计算过程。

## 推荐阅读

{% embed url="https://blog.csdn.net/qq\_40027052/article/details/78421155" %}

{% embed url="https://blog.csdn.net/u010041824/article/details/78855435" %}

{% embed url="https://pypi.org/project/keras-self-attention/" %}

{% embed url="https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/" %}

{% embed url="https://blog.csdn.net/xiaosongshine/article/details/90573585" %}

{% embed url="https://blog.csdn.net/xiaosongshine/article/details/90600028" %}

{% embed url="https://blog.csdn.net/u013608336/article/details/82792871" %}



