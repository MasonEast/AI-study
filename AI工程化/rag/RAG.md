# rag

## 大模型遇到的挑战

大语言模型（LLM）虽然展现出惊人的能力，但是也面对一些问题：

- 知识时效性：模型的训练数据存在截止日期，因此它无法回答最近（也就是LLM发布之后）发生的事情。
- 无法访问私有数据， 世界上大多数据都是私有数据，无法将其用于公开LLM的训练。

## 什么是RAG

RAG（Retrieval-Augmented Generation）是一种结合了检索和生成两种方法的模型，它能够解决大模型遇到的挑战。

RAG模型由三部分组成：

- Indexing：索引，用于存储知识库的索引。
- Retrieval：检索模块，用于从知识库中检索与输入相关的文档。
- Generation：生成模块，用于根据检索到的文档生成回答。

RAG模型的工作流程如下：

1. 输入一个查询，将其传递给检索模块。
2. 检索模块从知识库中检索与查询相关的文档。
3. 检索模块将检索到的文档传递给生成模块。
4. 生成模块根据检索到的文档生成回答。
5. 将生成的回答作为最终输出。

### 索引

我们有一些外部的文档要加载到系统中， Retriever 接收用户输入的问题 query ，目的是检索与输入问题相关的文档 documents。

这个过程中，自然会引发一个思考，如何建立 question 与 documents 之间的关联？通常是使用数字表示，将query和documents 都表示为一个数值向量，这样可以便于在大量示例中快速检索。

>为什么要将文档转换为数字表示？
>文本是离散的符号序列，相对于原始的文本，通常将文档转换为数字表示，比较数值向量之间的相似性要容易得多。
>索引是RAG模型的核心部分，它用于存储知识库的索引。索引可以是一个简单的列表，也可以是一个更复杂的结构，例如倒排索引或知识图谱。

#### 实现从文本到数值向量的转换

- 基于统计的方法 ：比如TF-IDF，词袋模型等方法，会将文档表示为一个稀疏向量。
- 基于机器学习的方法，word2vec，glove，BERT等方法，会将文本表示为一个稠密向量。

将文本转换为数值向量这一步之后，整个 Indexing 阶段的流程应该包括以下步骤：

- 文档加载， 文档的加载涉及将原始文档导入到系统的过程中，是整个indexing 的起点。
- 分块，由于文档最终要作为额外的上下文喂入LLM，要将文档长度其限制在LLM的下文窗口之内。因此，要对文档进行分块，也就是将大的文档切片为小的文本块。
- 嵌入， 前面已经说过，为了建立输入问题与文档之间的关系，要将其转换为数值向量。这一过程称为嵌入。
- 检索，在得到问题与文档的嵌入之后，可以使用不同的衡量向量相关性的方法，用于检索相关文档。检索这一过程将在下一节详述。

### 检索

 Indexing 阶段，就是将文档加载进来，分成小块，转换为易于搜索的数值向量形式，将其存储在向量数据库中。当给定一个输入的 query 时，我们会使用同样的嵌入方法 query 转换为向量，向量数据库会执行相似性搜索并返回与该问题相关的documents。

如果深入了解该过程，直觉就像是寻找某个点的邻居。
>举个例子，假设文档获得的嵌入只有3个维度，每个文档都会成为3D空间中的一个点。点的位置是根据 document 的语义或者内容决定的。
>位置相近的documents 拥有相似的语义或内容。我们将 query 做同样的嵌入，然后在 query 周围的空间去搜索。
>直观上来理解，就是哪些 documents 距离query 近， 这些documents与query具有相似的语义。

### 生成

拿到了与query 相关的文档之后， 接下来要做的就是“生成”， 将检索到的文档填充到LLM的上下文窗口中去，让LLM根据上下文生成最终的答案。
如何将检索与LLM连接起来呢？答案是prompt。 某种程度上，可以直接将prompt理解为具有占位符的一个模板，其中包含一些keys， 每个key 都是可以被填充的。

接下来要做的是：

1. 将query和检索到的documents作为输入，填充到prompt中。
2. 将填充后的prompt喂给LLM， 让LLM根据上下文生成最终的答案。

## 优化

### Query Translation

Query Translation 是一种优化方法，用于将用户的查询转换为更精确的查询。这种方法可以用于提高RAG模型的性能和准确性。

- 抽象化： 通过将具体问题抽象化，揭示问题的本质，也就是从更高的维度来看待问题。
- 具象化： 通过拆解复杂问题，让问题变得更具体。典型方法是将原始问题分解为多个有序的子问题，逐步解决，最终达成解决原始问题的任务。
- 重写： 从多个视角看待问题，同一个问题可能有100中写法，用不同的措辞来表达同一个问题。

#### Multi-Query（多重查询）

针对不同的角度看待一个问题，从不同的视角，可以生成多个不同表述的 query，分别去做检索，将检索到的所有文档汇总后，输入到LLM中。

- 基于原始的用户 query 生成多个query，分别是Q1, Q2,Q3 ，这些生成的query是不同角度对原始 query 的补充。
- 对形成的每个query，Q1, Q2,Q3 ，都去检索一批相关文档。
- 所有的相关文档都会被喂给LLM，这样LLM就会生成比较完整和全面的答案

#### RAG Fustion（RAG融合）

RAG-fustion是在multi-query基础上，对检索进行重新排序，输出top k 个最相关文档。

RAG-Fustion 使用了Reciprocal Rank Fusion 倒排相关性 (RRF) 用于文档的重新排序。

原理：如果一个文档被多个查询同时命中，则排名提升（类似投票机制）。
公式：​

（n=查询数量，k=平滑常数，rank(d,q)=文档d在第q个查询中的排名）
一句话总结，RAG-Fusion 是 Multi-Query 的“智能版”——它不只追求检索数量，而是通过投票加权+去重过滤，进一步提升检索质量

#### Decompsition（分解）

将query分解为子问题。

这部分有两个相关的工作，一个是来自于Google 的 “Least-to-most” ， 主要流程包括：

- 将原始query分解为多个子问题
- 按逻辑顺序解决每个子问题，每一步都利用前面子问题的答案作为上下文，合成最终答案。

#### Step-back Prompting （后退提示）

Step-back prompting 包括两步：

- 抽象： 通过prompt LLM从原始问题中抽象出更高层次的概念或原理。这一步骤的目的是让模型理解问题的本质，从而更容易找到相关的信息和事实。
- 推理：其次，基于高层次概念或原理的事实，LLMs进行推理以解决原始问题。这一步骤的目的是利用高层次概念来指导低层次细节的推理过程，从而减少中间推理步骤中的错误

#### HyDE （假说文档嵌入）

HyDE (Hypothetical Document Embedding, 假说文档嵌入)同样是将具体的问题变得更抽象的一种方法。
回顾基础的RAG流程，将初始的query与documents 使用到同一embedding方法嵌入到同一特征空间中去，然后基于二者的相似性，检索到相关的文档，让LLM根据上下文回答问题。
然而， 这里存在的问题是：query 和 document 是两种不同的文本对象，一般来说，document 来自各种不同的来源，可以非常长，但是query 通常是非常简短的。这就引发一个思考，如果将query转换为一个document，然后再做嵌入不就行了吗？
原理解析
HyDE就是这样出现的, 具体来说， 包括以下步骤：

- 生成假设文档： 给定一个 query, 让模型针对其生成一个假设文档。这个文档的目的在于捕捉与 query 相关的模式，它不是真实的文档，可能包含一些错误的信息。
- 编码假设文档： 将生成的假设文档编码为一个嵌入向量
- 检索真实文档： 使用假设文档的嵌入向量在预料库中进行检索，基于向量相似性，检索到与假设文档相似的真实文档。



## RAG的优缺点

RAG模型具有以下优点：

- 知识时效性：由于RAG模型可以从私有数据中检索信息，因此它可以回答最近发生的事情。
- 知识丰富性：由于RAG模型可以从私有数据中检索信息，因此它可以回答更复杂的问题。
- 知识个性化：由于RAG模型可以从私有数据中检索信息，因此它可以回答更个性化的问题。

RAG模型具有以下缺点：

- 检索速度：由于RAG模型需要从知识库中检索信息，因此它的检索速度可能会受到限制。
