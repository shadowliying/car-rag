# coding=utf-8
import json  # JSON格式处理
import sys   # 系统参数交互
import re    # 正则表达式
import numpy as np
from text2vec import SentenceModel, semantic_search, Similarity   # 句子编码模型   # 语义搜索  # 相似度计算


simModel_path = './pre_train_model/text2vec-base-chinese'  # 相似度模型路径
simModel = SentenceModel(model_name_or_path=simModel_path, device='cuda:0')  # 模型加载

# Jaccard相似度函数
def calc_jaccard(list_a, list_b, threshold=0.3):
    size_a, size_b = len(list_a), len(list_b)
    list_c = [i for i in list_a if i in list_b]  # 计算交集
    size_c = len(list_c)
    score = size_c / (size_b + 1e-6)
    if score > threshold:   # 阈值判定
        return 1
    else:
        return 0


def report_score(gold_path, predict_path):  # gold_path 标准答案文件路径  predict_path 预测答案文件路径
    # 加载标准答案数据和预测答案数据
    gold_info = json.load(open(gold_path))     # 从标准答案文件中加载数据，存储为字典列表
    pred_info = json.load(open(predict_path))  # 从预测答案文件中加载数据，存储为字典列表

    idx = 0  # 用于追踪当前处理的数据索引
    for gold, pred in zip(gold_info, pred_info):  # 同时遍历标准答案和预测答案
        question = gold["question"]    # 提取问题文本
        keywords = gold["keywords"]    # 提取标准答案中的关键词列表
        gold = gold["answer"].strip()  # 提取并去除标准答案的首尾空格
        pred = pred["answer_4"].strip()   # 提取并去除预测答案的首尾空格（字段需对齐）

        # 判断标准答案和预测答案是否为"无答案"情况
        if gold == "无答案" and pred != gold:
            score = 0.0  # 如果标准答案为"无答案"但预测答案不一致，则得分为0
        elif gold == "无答案" and pred == gold:
            score = 1.0  # 如果标准答案和预测答案均为"无答案"，则得分为1
        else:
            # 计算预测答案的语义相似度分数
            semantic_score = semantic_search(simModel.encode([gold]), simModel.encode(pred), top_k=1)[0][0]['score'] 
            # 提取预测答案中与标准答案关键词匹配的关键词
            join_keywords = [word for word in keywords if word in pred]
            # 计算关键词匹配的Jaccard相似度分数
            keyword_score = calc_jaccard(join_keywords, keywords)
            # 计算最终得分，语义相似度和关键词相似度各占50%
            score = 0.5 * keyword_score + 0.5 * semantic_score   

        # 将计算的得分和预测答案存储到标准答案数据中
        gold_info[idx]["score"] = score
        gold_info[idx]["predict"] = pred 
        idx += 1  # 更新索引
        # 打印当前问题的预测结果和得分
        print(f"预测: {question}, 得分: {score}")

    return gold_info  # 返回包含得分和预测答案的标准答案数据


if __name__ == "__main__":
    '''
      online evaluation
    '''

    # 标准答案路径
    gold_path = "./data/gold2.json" 
    print("Read gold from %s" % gold_path)

    # 预测文件路径
    predict_path = "./data/result.json" 
    print("Read predict file from %s" % predict_path)

    results = report_score(gold_path, predict_path)

    # 输出最终得分
    final_score = np.mean([item["score"] for item in results])
    print("\n")
    print("="*100)
    print(f"预测问题数：{len(results)}, 预测最终得分：{final_score}")
    print("="*100)

    # 结果文件路径
    metric_path = "./data/metrics.json" 
    results_info = json.dumps(results, ensure_ascii=False, indent=2)
    with open(metric_path, "w") as fd:
        fd.write(results_info)
    print(f"\n结果文件保存至{metric_path}")

