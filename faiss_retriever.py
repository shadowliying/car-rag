#!/usr/bin/env python
# coding: utf-8


from langchain.schema import Document   # 用于结构化存储文本块
from langchain.vectorstores import Chroma,FAISS  # 向量数据库 用于存储文本嵌入（向量）并支持高效相似性搜索
from langchain.embeddings.huggingface import HuggingFaceEmbeddings # 使用HuggingFace模型将文本转换为嵌入向量
from pdf_parse import DataProcess   # PDF提取文本类
import torch


class FaissRetriever(object):
    # 初始化文档块索引，然后插入faiss库
    def __init__(self, model_path, data):
        self.embeddings  = HuggingFaceEmbeddings(
                               model_name = model_path,  # 模型路径
                               model_kwargs = {"device":"cuda"},  # 指定模型运行在GPU上
                               encode_kwargs = {"batch_size": 64}  # 批量处理文本，提高计算效率
                               # model_kwargs = {"device":"cuda:1"}
                           )
        # 2. 把文本转成Document对象
        docs = []
        for idx, line in enumerate(data):  # 遍历每一行文本
            line = line.strip("\n").strip()  # 去除首尾换行符和空格
            words = line.split("\t")     # 按制表符分割
            docs.append(Document(page_content=words[0], metadata={"id": idx}))  # 存储文档的文本内容。 存储与文档相关的元数据
        # 3. 构建FAISS向量索引
        self.vector_store = FAISS.from_documents(docs, self.embeddings)  # 将 Document 对象转换为向量，并构建FAISS索引
        
        # 生成embeddings的时间较长，跑完第一次可以把结果持久化，后面直接load
        #self.vector_store.save_local("./faiss_index")
        #self.vector_store = FAISS.load_local("./faiss_index", self.embeddings, allow_dangerous_deserialization=True)
        
        # 删除嵌入模型并清理GPU显存
        del self.embeddings
        torch.cuda.empty_cache()

    # 获取top-K分数最高的文档块
    def GetTopK(self, query, k):
       context = self.vector_store.similarity_search_with_score(query, k=k) #从向量数据库中检索相似度最高的前 k 个文档块，并返回文档内容及其相似度分数
       return context

    # 返回faiss向量检索对象
    def GetvectorStore(self):   # 返回类内部的FAISS向量存储对象 self.vector_store，允许外部直接操作向量数据库。
        return self.vector_store

if __name__ == "__main__":
    base = "."
    model_name = base + "/pre_train_model/m3e-large" #text2vec-large-chinese
    dp =  DataProcess(pdf_path = base + "/data/train_a.pdf")
    dp.ParseBlock(max_seq = 1024)
    dp.ParseBlock(max_seq = 512)
    print(len(dp.data))
    dp.ParseAllPage(max_seq = 256)
    dp.ParseAllPage(max_seq = 512)
    print(len(dp.data))
    dp.ParseOnePageWithRule(max_seq = 256)
    dp.ParseOnePageWithRule(max_seq = 512)
    print(len(dp.data))
    data = dp.data

    faissretriever = FaissRetriever(model_name, data)
    faiss_ans = faissretriever.GetTopK("如何预防新冠肺炎", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("交通事故如何处理", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("吉利集团的董事长是谁", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("吉利汽车语音组手叫什么", 6)
    print(faiss_ans)
