---
title: "News_深度学习BERT"
date: 2020-08-04T23:10:23+08:00
lastmod: 2020-08-04T23:50:23+08:00
draft: false
tags: ["preview", "NLP", "tag-5"]
categories: ["NLP"]

toc: false

---

## Transformer

Transformer是一种**完全基于Attention机制来加速深度学习训练过程的算法模型**，其最大的优势在于其在并行化处理上做出的贡献。换句话说，Transformer就是一个**带有self-attention机制的seq2seq 模型**，即输入是一个sequence，输出也是一个sequence的模型。如下图所示：

![img](https://img-blog.csdnimg.cn/20190729105442326.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L09zY2FyNjI4MDg2OA==,size_16,color_FFFFFF,t_70)

**self-attention的架构**

假设有x1、x2、x3、x4x1、x2、x3、x4*x*1、*x*2、*x*3、*x*4四个序列，首先进行带权乘法ai=Wxiai = W \, xi*a**i*=*W**x**i*，得到新的序列a1、a2、a3、a4a1、a2、a3、a4*a*1、*a*2、*a*3、*a*4。

![img](https://img-blog.csdnimg.cn/20190729122430138.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L09zY2FyNjI4MDg2OA==,size_16,color_FFFFFF,t_70)

然后将aiai*a**i*分别乘以三个不同的权重矩阵得到qi、ki、viqi、ki、vi*q**i*、*k**i*、*v**i*三个向量，qi=Wqaiqi=Wq \, ai*q**i*=*W**q**a**i*，ki=Wqaiki=Wq \, ai*k**i*=*W**q**a**i*，vi=Wvaivi=Wv \, ai*v**i*=*W**v**a**i*，qq*q*表示的是query，需要match其他的向量，kk*k*表示的是key，是需要被qq*q*来match的，vv*v*表示value，表示需要被抽取出来的信息。

![img](https://img-blog.csdnimg.cn/20190729152314974.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L09zY2FyNjI4MDg2OA==,size_16,color_FFFFFF,t_70)

接下来让每一个qq*q*对每一个kk*k*做attention操作。attention操作的目的是输入两个向量，输出一个数，这里使用scaled点积来实现。q1q1*q*1和kiki*k**i*做attention得到α1,i{\alpha}_{1,i}*α*1,*i*，

α1,i=q1⋅ki/d‾‾√{\alpha}_{1,i} = q1 \cdot ki / \sqrt{d}*α*1,*i*=*q*1⋅*k**i*/*d*

其中dd*d*表示qq*q*和vv*v*的维数。

![img](https://img-blog.csdnimg.cn/20190729154406562.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L09zY2FyNjI4MDg2OA==,size_16,color_FFFFFF,t_70)

最后将α1,i{\alpha}_{1,i}*α*1,*i*带入softmax函数，写成矩阵形式即为：

Attention(Q,K,V)=softmax(QKTd√V)Attention(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt{d}} V \right)*A**t**t**e**n**t**i**o**n*(*Q*,*K*,*V*)=*s**o**f**t**m**a**x*(*d**Q**K**T**V*)

![img](https://img-blog.csdnimg.cn/20190729155856638.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L09zY2FyNjI4MDg2OA==,size_16,color_FFFFFF,t_70)

然后求出$b1 = \sum \hat{\alpha}_{1,i} \cdot vi $。整个过程如下图所示：

![img](https://img-blog.csdnimg.cn/20190729160738214.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L09zY2FyNjI4MDg2OA==,size_16,color_FFFFFF,t_70)

由上图可知，求b1b1*b*1的时候需要用到序列中的所有值a1,a2,a3,a4a1,a2,a3,a4*a*1,*a*2,*a*3,*a*4，但是对序列的每个部分的关注程度有所不同，通过改变αˆ1,i\hat{\alpha}_{1,i}*α*^1,*i*前的权重vivi*v**i*可以对序列的每一部分赋予不同的关注度，对重点关注的部分赋予较大的权重，不太关注的部分赋予较小的权重。
用同样的方法可以求出b2,b3,b4b2,b3,b4*b*2,*b*3,*b*4，而且b1,b2,b3,b4b1,b2,b3,b4*b*1,*b*2,*b*3,*b*4是平行地被计算出来的，相互之间没有影响。整个过程可以看作是一个self-attention layer，输入x1,x2,x3,x4x1,x2,x3,x4*x*1,*x*2,*x*3,*x*4，输出b1,b2,b3,b4b1,b2,b3,b4*b*1,*b*2,*b*3,*b*4.

Transformer所使用的注意力机制的核心思想是去**计算一句话中的每个词对于这句话中所有词的相互关系**，然后认为**这些词与词之间的相互关系在一定程度上反应了这句话中不同词之间的关联性以及重要程度**。因此再利用这些相互关系来调整每个词的重要性（权重）就可以获得每个词新的表达。这个新的表征不但蕴含了该词本身，还蕴含了其他词与这个词的关系，因此和单纯的词向量相比是一个更加全局的表达。使用了Attention机制，将序列中的任意两个位置之间的距离缩小为一个常量。

![img](https://img-blog.csdnimg.cn/20190729162622945.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L09zY2FyNjI4MDg2OA==,size_16,color_FFFFFF,t_70)

乘积部分的表示如下图：

![img](https://img-blog.csdnimg.cn/20190729170755467.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L09zY2FyNjI4MDg2OA==,size_16,color_FFFFFF,t_70)

整个transformer的结构如下图：

![img](http://5b0988e595225.cdn.sohucs.com/images/20190725/4b086c3175234abbb686a0cebe6012ce.jpeg)

左边的部分是encoder，右边的部分是decoder。

encoder部分的步骤如下所示：

- 对input进行**embedding**操作，将单词表示成长度为embedding size的向量

- 对embedding之后的词向量进行**positional encoding**，即在生成q、k、vq、k、v*q*、*k*、*v*的时候，给每一个aiai*a**i*加上一个相同维数的向量eiei*e**i*，如下图：

  ![img](https://img-blog.csdnimg.cn/20190729223129293.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L09zY2FyNjI4MDg2OA==,size_16,color_FFFFFF,t_70)

  eiei*e**i*表示了位置的信息，每一个位置对应不同的eiei*e**i*，id为p的位置对应的位置向量的公式为：

  e2i=sin(p/100002i/d)e_{2i} = sin \left( p/10000^{2i/d} \right)*e*2*i*=*s**i**n*(*p*/100002*i*/*d*)

  e2i+1=cos(p/100002i/d)e_{2i+1} = cos \left( p/10000^{2i/d} \right)*e*2*i*+1=*c**o**s*(*p*/100002*i*/*d*)

  对于NLP中的任务来说，顺序是很重要的信息，它代表着局部甚至是全局的结构，学习不到顺序信息，那么效果将会大打折扣。通过结合位置向量和词向量，给每个词都引入了一定的位置信息，这样Attention就可以分辨出不同位置的词

- 进行**muti-head attention**操作，同时生成多个q、k、vq、k、v*q*、*k*、*v*分别进行attention（参数不同），然后把结果拼接起来

- **add & norm**操作，将muti-head attention的input和output进行相加，然后进行layer normalization操作

  layer normalization(LN) 和batch normalization(BN) 的过程相反，BN表示在一个batch里面的数据相同的维度进行normalization，而LN表示在对每一个数据所有的维度进行normalization操作。假设数据的规模是10行3列，即batchsize = 10，每一行数据有三个特征，假设这三个特征是 [身高、体重、年龄]。那么BN是针对每一列（特征）进行缩放，例如算出[身高]的均值与方差，再对身高这一列的10个数据进行缩放。体重和年龄同理。这是一种“**列缩放**”。而layer方向相反，它针对的是每一行进行缩放。即只看一条记录，算出这条记录所有特征的均值与方差再缩放。这是一种“**行缩放**”

- 然后进行一个**feed forward**和**add & norm**操作，feed forward会对每一个输入的序列进行操作，再进行一个add & norm操作

至此encoder的操作就完成了，接下来看decoder操作：

- 进行**positional encoding**
- **masked muti-head attention**操作，即对之前产生的序列进行attention
- 进行**add & norm**操作
- **将encoder的输出和上一轮add & norm操作的结构进行muti-head attention和add & norm操作**
- **feed forward 和 add & norm**，整个的过程可以重复N次
- 经过**线性层和softmax层**得到最终输出



## Bert

Bert(Bidirectional Encoder Representation form Transformers)，即双向Transformer的Encoder，其中“双向”表示模型在处理某一个词时，它能同时利用前面的词和后面的词两部分信息。Bert的模型架构基于多层双向转换解码，通过执行一系列预训练，进而得到深层的上下文表示。Bert的基本思想和Word2Vec、CBOW一样，都是给定context，来预测下一个词。BERT的结构和ELMo相似都是双向结构。BERT模型结构如下图所示：

![img](https://camo.githubusercontent.com/f5297c1c8c1e71180cb62aca463503d21122e911/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303731343231313331363136372e706e67)

Bert的实现分为两个阶段：第一个阶段叫做：**Pre-training**，跟WordEmbedding类似，利用现有无标记的语料训练一个语言模型。第二个阶段叫做：**Fine-tuning**，利用预训练好的语言模型，完成具体的NLP下游任务。

**Bert的输入：**

Bert的输入包含三个部分：**Token Embeddings、Segment Embeddings、Position Embeddings**。这三个部分在整个过程中是可以学习的。

![img](https://camo.githubusercontent.com/1fa48883f724ed604556d3bb9d241511012d1972/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303731343231313334383435362e706e67)

- CLS：Classification Token，用来区分不同的类
- SEP：Special Token，用来分隔两个不同的句子
- Token Embedding：对输入的单词进行Embedding
- Segment Embedding：标记输入的单词是属于句子A还是句子B
- Position Embedding：标记每一个Token的位置

**Pre-training：**

Bert的预训练有两个任务：**Masked Language Model（MLM）\**和\**Next Sentence Predicition（NSP）**。在训练Bert的时候，这两个任务是同时训练的，Bert的损失函数是把这两个任务的损失函数加起来的，是一个多任务训练。

Masked Language Model的灵感来源于完形填空，将15%的Tokens掩盖。被掩盖的15%的Tokens又分为三种情况：80%的字符用字符“MASK”替换，10%的字符用另外的字符替换；10%的字符是保持不动。然后模型尝试基于序列中其他未被掩盖的单词的上下文来预测被掩盖的原单词。最后在计算损失时，只计算被掩盖的15%的Tokens。

Next Sentence Prediction，即给出两个句子A和B，B有一半的可能性是A的下一句话，训练模型来预测B是不是A的下一句话。通过训练，使模型具备理解长序列上下文的联系的能力。

**Fine-tuning：**

![img](https://camo.githubusercontent.com/265b6f273c9731886f63f66b7beb2805226cee73/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303731343231313430393538322e706e67)

- 分类任务：输入端，可以是句子A和句子B也可以是单一句子，CLS后接softmax用于分类
- 答案抽取，比如SQuAd v1.0，训练一个start和end向量分别为S，E，bert的每个输出向量和S或E计算dot product，之后对所有输出节点的点乘结果进行softmax得到该节点对应位置的起始概率或者或终止概率， 假设Bert的输出向量为TT*T*，则用S⋅Ti+E⋅TjS·Ti + E·Tj*S*⋅*T**i*+*E*⋅*T**j*表示从ii*i*位置起始，jj*j*位置终止的概率，最大的概率对应ii*i*和j(i<j)j(i<j)*j*(*i*<*j*)即为预测的answer span的起点终点，训练的目标是最大化正确的起始,终点的概率
- SQuAD v2.0和SQuAD 1.1的区别在于可以有答案不存在给定文本中，因此增加CLS的节点输出为CC*C*，当最大的分数对应i,ji,j*i*,*j*所在的CLS的时候，即S⋅Ti+E⋅TjS·Ti + E·Tj*S*⋅*T**i*+*E*⋅*T**j*的最大值小于S⋅C+E⋅CS·C + E·C*S*⋅*C*+*E*⋅*C*时，不存在答案

## 代码实现

初始化：

![屏幕快照 2020-08-05 上午12.18.21](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.18.21.png)

读取数据并划分验证集：

![屏幕快照 2020-08-05 上午12.20.45](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.20.45.png)

![屏幕快照 2020-08-05 上午12.20.56](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.20.56.png)

![屏幕快照 2020-08-05 上午12.21.08](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.21.08.png)

建立训练，验证和测试集：

![屏幕快照 2020-08-05 上午12.22.35](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.22.35.png)

建立词典：

![屏幕快照 2020-08-05 上午12.23.17](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.23.17.png)

![屏幕快照 2020-08-05 上午12.23.27](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.23.27.png)

![屏幕快照 2020-08-05 上午12.23.36](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.23.36.png)

![屏幕快照 2020-08-05 上午12.23.46](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.23.46.png)

建立各个模块：

![屏幕快照 2020-08-05 上午12.24.51](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.24.51.png)

![屏幕快照 2020-08-05 上午12.25.04](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.25.04.png)

![屏幕快照 2020-08-05 上午12.25.17](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.25.17.png)

![屏幕快照 2020-08-05 上午12.25.29](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.25.29.png)

![屏幕快照 2020-08-05 上午12.25.45](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.25.45.png)

![屏幕快照 2020-08-05 上午12.26.51](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.26.51.png)

![屏幕快照 2020-08-05 上午12.27.01](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.27.01.png)

建立优化器：

![屏幕快照 2020-08-05 上午12.27.48](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.27.48.png)

![屏幕快照 2020-08-05 上午12.27.58](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.27.58.png)

建立数据库：

![屏幕快照 2020-08-05 上午12.28.39](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.28.39.png)

![屏幕快照 2020-08-05 上午12.28.51](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.28.51.png)

划分batch：

![屏幕快照 2020-08-05 上午12.30.03](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.30.03.png)

![屏幕快照 2020-08-05 上午12.30.13](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.30.13.png)

其他用到的函数：

![屏幕快照 2020-08-05 上午12.31.04](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.31.04.png)

#### 训练函数

建立训练器：

![屏幕快照 2020-08-05 上午12.32.11](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.32.11.png)

![屏幕快照 2020-08-05 上午12.32.20](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.32.20.png)

![屏幕快照 2020-08-05 上午12.32.30](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.32.30.png)

![屏幕快照 2020-08-05 上午12.32.40](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.32.40.png)

![屏幕快照 2020-08-05 上午12.32.48](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.32.48.png)

![屏幕快照 2020-08-05 上午12.32.59](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.32.59.png)

![屏幕快照 2020-08-05 上午12.33.08](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.33.08.png)

![屏幕快照 2020-08-05 上午12.33.19](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.33.19.png)

进行训练：

![屏幕快照 2020-08-05 上午12.33.57](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.33.57.png)

训练和测试结果：

![屏幕快照 2020-08-05 上午12.37.42](/Users/lishanshan/Desktop/屏幕快照 2020-08-05 上午12.37.42.png)

