#!/usr/bin/env python
# coding: utf-8


from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from pdf_parse import DataProcess
import jieba

class BM25(object):

    # 遍历文档，首先做分词，然后把分词后的文档和全文文档建立索引和映射关系 
    def __init__(self, documents):

        docs = []  # 存储分词后的文档
        full_docs = []  # 存储原始格式文档
        for idx, line in enumerate(documents): # 遍历输入的文档集合并添加索引
            # 清洗数据：去除首尾换行符和空格
            line = line.strip("\n").strip()
            # 过滤短文本（长度小于5的跳过）
            if(len(line)<5):
                continue
            # 中文分词处理：使用jieba的搜索引擎模式分词。  例："自然语言处理" -> "自然 语言 处理"
            tokens = " ".join(jieba.cut_for_search(line))
            # docs.append(Document(page_content=tokens, metadata={"id": idx, "cate":words[1],"pageid":words[2]}))
            docs.append(Document(page_content=tokens, metadata={"id": idx})) # 构建带分词的文档对象
            # full_docs.append(Document(page_content=words[0], metadata={"id": idx, "cate":words[1], "pageid":words[2]}))
            words = line.split("\t")
            full_docs.append(Document(page_content=words[0], metadata={"id": idx}))
        self.documents = docs
        self.full_documents = full_docs
        self.retriever = self._init_bm25()  # 初始化BM25检索器

    # 初始化BM25的知识库
    def _init_bm25(self):
        return BM25Retriever.from_documents(self.documents)

    # 获得得分在topk的文档和分数
    def GetBM25TopK(self, query, topk):
        self.retriever.k = topk
        query = " ".join(jieba.cut_for_search(query))
        ans_docs = self.retriever.get_relevant_documents(query)  # 调用 BM25 检索器的文档查询方法
        ans = []
        for line in ans_docs:
            ans.append(self.full_documents[line.metadata["id"]])
        return ans

if __name__ == "__main__":

    # bm2.5
    dp =  DataProcess(pdf_path = "./data/train_a.pdf")
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
    bm25 = BM25(data)
    res = bm25.GetBM25TopK("座椅加热", 6)
    print(res)
