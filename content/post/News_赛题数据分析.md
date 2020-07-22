---
title: "News_赛题数据分析"
date: 2020-07-21T23:10:23+08:00
lastmod: 2020-07-21T23:50:23+08:00
draft: false
tags: ["preview", "NLP", "tag-5"]
categories: ["NLP"]

toc: false

---


本节是通过数据分析来对天池入门赛数据做一些学习。

#### 一. 目标：

1. 学会用pandas读取数据，plt展示图像，collections统计分布规律



#### 二. 数据读取：

```
import pandas as pd
path_data='/Users/lishanshan/Workspace/Datawhale/NLP/train_set.csv'

df = pd.read_csv(path_data, sep='\t')
df.head()
```

![屏幕快照 2020-07-22 下午10.10.16](/Users/lishanshan/Desktop/屏幕快照 2020-07-22 下午10.10.16.png)

第一列为label值，第二列text是新闻内容。

#### 三. 数据分析：

##### 3.1 目标

1）新闻文本内容长度是多少？

2）类别（label）是什么样的，什么类别占比比较多？

3）字符在全局的分布是怎样的？

##### 3.2 句子长度分析

​	句子中的单词用空格隔开，所以可以直接统计单词的个数来统计句子长度。

```
df['text_len'] = df['text'].apply(lambda x: len(x.split(' ')))
print(df['text_len'].describe())
```

![屏幕快照 2020-07-22 下午11.32.36](/Users/lishanshan/Desktop/屏幕快照 2020-07-22 下午11.32.36.png)

总共有20000行句子，最长的有50000+单词，最短的只有2个，有一半的新闻长度在700以内。

所以我们导入matplot库来看句子长度的分布，

```
_ = plt.hist(df['text_len'], bins=200)

plt.xlabel('Text char count')
plt.title("Histogram of char count")
```



![屏幕快照 2020-07-22 下午11.35.06](/Users/lishanshan/Desktop/屏幕快照 2020-07-22 下午11.35.06.png)



##### 3.3 字符分布统计

我们可以通过collections库中的Counter函数来统计词频

```
from collections import Counter
all_lines = ' '.join(list(df['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse=True)
print(len(word_count))
print(word_count[0])
print(word_count[-1])
```

![屏幕快照 2020-07-22 下午11.39.07](/Users/lishanshan/Desktop/屏幕快照 2020-07-22 下午11.39.07.png)

从统计结果可以看出，总共有6869个字符，其中'3750'这个字符出现最多，有7482224次，而'3133'这个字符出现最少，只有1次。

接下来下面代码统计了不同字符在句子中出现的次数，其中字符3750，字符900和字符648在20w新闻的覆盖率接近99%，很有可能是标点符号。

```
df['text_unique'] = df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
all_lines = ' '.join(list(df['text_unique']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:int(d[1]), reverse = True)

print(word_count[0])
print(word_count[1])
print(word_count[2])
```

![屏幕快照 2020-07-22 下午11.40.27](/Users/lishanshan/Desktop/屏幕快照 2020-07-22 下午11.40.27.png)

##### 四. 结论：

1. 赛题中每个新闻包含的字符个数平均为1000个，有少数的新闻比较长。
2. 赛题总共包括7000-8000个字符。
3. 每个新闻平均字符个数较多，可能需要截断。
4. 由于类别不均衡，会严重影响模型的精度。
