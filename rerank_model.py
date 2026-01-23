# 导入Hugging Face Transformers库组件 
# - AutoModelForSequenceClassification: 自动选择适合文本分类任务的模型架构
# - AutoTokenizer: 自动匹配模型对应的文本处理器
from transformers import AutoModelForSequenceClassification, AutoTokenizer   
import os
import torch

# 导入自定义BM25检索器
from bm25_retriever import BM25
from pdf_parse import DataProcess
from config import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


# 释放gpu上没有用到的显存以及显存碎片
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# 加载rerank模型
class reRankLLM(object):
    def __init__(self, model_path, max_length = 512):
        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 加载预训练序列分类模型
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # 设置模型为评估模式（关闭dropout等训练层）
        self.model.eval()
        # 将模型转换为半精度（FP16）以节省显存
        self.model.half()
        # 将模型移至默认GPU（等效于.to("cuda")）
        self.model.cuda()
        # self.model.to(torch.device("cuda:1"))
        
        # 设置最大序列长度
        self.max_length = max_length

    # 输入文档对，返回每一对(query, doc)的相关得分，并从大到小排序
    def predict(self, query, docs):
        pairs = [(query, doc.page_content) for doc in docs]
        # inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length).to("cuda:1")

        # 批量编码输入文本（自动处理padding和截断）
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length).to("cuda")
        # 禁用梯度计算（推理阶段）
        with torch.no_grad():
            # 前向传播获取原始输出
            scores = self.model(**inputs).logits

        # detach() 将张量 scores 从计算图中分离出来，返回一个新的张量，该张量不带梯度信息。
        # cpu() 将张量从 GPU（如果有）移动到 CPU。
        # clone() 创建张量的一个副本
        # numpy() 将张量转换为 NumPy 数组。
        scores = scores.detach().cpu().clone().numpy()
        # 按分数降序排列文档
        response = [doc for score, doc in sorted(
                    zip(scores, docs), 
                    reverse=True, 
                    key=lambda x:x[0])]
        # 执行显存回收
        torch_gc()
        return response

if __name__ == "__main__":
    bge_reranker_large = "./pre_train_model/bge-reranker-large"
    rerank = reRankLLM(bge_reranker_large)
