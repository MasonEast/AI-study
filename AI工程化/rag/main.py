

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

import bs4 # HTML解析库，用于处理网页结构
from langchain_community.document_loaders import WebBaseLoader

# from dotenv import load_dotenv

# load_dotenv()  # 自动读取.env文件

# **************************** 索引化流程 ****************************
# 1. 加载外部文档
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
            )
    ),
)
blog_docs = loader.load()

print("加载的文档数量:------", len(blog_docs))

# 2. 分块， 
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

splits = text_splitter.split_documents(blog_docs)


# 3. 创建向量存储


# 使用Chroma作为向量存储
vectorstore = Chroma.from_documents(documents=splits, embedding=ZhipuAIEmbeddings(model="embedding-3", api_key="c8b19f36d9634db2b7eb0575ba61c16b.oiHdRizgXqPJr2GT"))
    
# 将向量存储转换为检索器
retriever = vectorstore.as_retriever()

# **************************** 检索流程 ****************************
# 4. 检索相关文档
docs = retriever.get_relevant_documents("What is Task Decomposition?")

# ***************************** 生成流程 ****************************

    
# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
# 定义 prompt 模板
prompt = ChatPromptTemplate.from_template(template)

# 定义 LLM
llm = ChatOpenAI(
    model="glm-4",
    temperature=0.3,
    max_tokens=200,
    api_key="9908930230da402bb58cb5511d253230.OgtuJwgII4yK8Wd7",
    base_url='https://open.bigmodel.cn/api/paas/v4/',
)


# 定义Chain
# 在LangChain 中，有一个名为LCEL的表达式语言，它允许您以一种简洁的方式定义数据处理流程。
chain = prompt | llm

# 调用
response = chain.invoke({"context":docs,"question":"What is Task Decomposition?"})



# messages = [
#     SystemMessage(content="你是一名精通了 golang 的专家"),
#     HumanMessage(content="写一个  golang 的 hello world 程序"),
# ]

# response = chat.invoke(messages)

print(response.content)
