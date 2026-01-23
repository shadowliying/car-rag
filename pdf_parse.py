#!/usr/bin/env python
# coding: utf-8

import pdfplumber  # 用于处理PDF文件  它提供了简单而强大的接口，使得从PDF文档中提取文本、表格和其他元素变得更加容易。
from PyPDF2 import PdfReader   # 读取 PDF 文件  提取文本  获取元数据   处理页面（旋转，合并，拆分）
 

class DataProcess(object):

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.data = []

    # 滑动窗口功能实现，其中fast代表当前遍历句子的index，slow代表每次窗口开始滑动的起点。默认窗口直接滑动的overlap是1个句子。
    '''
    sentences: 输入的句子列表（如 ["句子1", "句子2", "句子3"]）。
    kernel=512: 窗口的最大长度（单位：字符数），默认 512。
    stride=1: 窗口滑动步长（单位：句子数），默认每次滑动 1 个句子（即重叠 kernel-1 个句子）。
    保存的都是超出长度的文本块。
    '''
    def SlidingWindow(self, sentences, kernel = 512, stride = 1):
        sz = len(sentences) # 总句子数
        cur = ""    #  当前窗口累积的文本
        fast = 0    # 快指针 （当前处理到的句子索引）   
        slow = 0    # 慢指针（窗口起始句子索引）   fast 和 slow 构成滑动窗口的左右边界。
        while(fast < len(sentences)):
            sentence = sentences[fast]   # 当前处理的句子
            if(len(cur + sentence) > kernel and (cur + sentence) not in self.data):  # 检查当前累积的句子是否超出窗口长度且内容未重复
                self.data.append(cur + sentence + "。")   # 保存当前窗口内容
                cur = cur[len(sentences[slow] + "。"):]   # 移除最旧的句子
                slow = slow + 1                           # 窗口左边界右移
            cur = cur + sentence + "。"                   # 将当前句子加入窗口
            fast = fast + 1                               # 处理下一个句子

    
    '''
    该方法用于过滤和预处理文本数据，主要功能包括：
        1.过滤过短文本（长度 < 6）。
        2.对超长文本（长度 > max_seq）按特定分隔符分割。
        3.清理文本中的换行符、逗号、制表符。
        4.去重后存储到 self.data 列表。
    line: 输入的文本行（字符串）。
    max_seq: 文本最大允许长度（默认 1024）。
    '''
    #  数据过滤，根据当前的文档内容的item划分句子，然后根据max_seq划分文档块。
    def Datafilter(self, line, header, pageid, max_seq = 1024):

         sz = len(line)  # 过滤过短文本（长度 < 6）。
         if(sz < 6):
             return 
 
         if(sz > max_seq):   # 处理超长文本（长度 > max_seq）
             # 根据特定分隔符分割文本
             if("■" in line):    # 传过来的数据可能是按标题块进行存储的，就按照下面可能的符号进行分割。
                 sentences = line.split("■")
             elif("•" in line):
                 sentences = line.split("•")
             elif("\t" in line):
                 sentences = line.split("\t")
             else:
                 sentences = line.split("。")
             # 处理分割后的子句
             for subsentence in sentences:
                 subsentence = subsentence.replace("\n", "")  # 清理换行符
                # 保留长度在 6 到 max_seq 之间的子句。
                 if(len(subsentence) < max_seq and len(subsentence) > 5):  
                     # 清理符号：移除逗号、换行符、制表符。
                     subsentence = subsentence.replace(",", "").replace("\n","").replace("\t","")  # 
                     if(subsentence not in self.data):
                         # 若子句不在 self.data 中，则添加。
                         self.data.append(subsentence)
         else: # 处理短文本（长度 ≤ max_seq）
             line = line.replace("\n","").replace(",", "").replace("\t","")  # 清理符号：移除逗号、换行符、制表符。
             if(line not in self.data):
                 self.data.append(line)

    '''
    方法用于从页面（如 PDF 文档的一页）中提取页眉文本
        1.跳过目录页：若页面包含“目录”或连续句点（".........."），则返回 None。
        2.位页眉位置：优先选择页面顶部区域（垂直坐标 top 在 17–20 范围内）的文本作为页眉。
        3.兜底策略：若未找到符合条件的页眉，则返回页面第一行文本。
    '''
    # 提取页头即一级标题
    def GetHeader(self, page):
        try:
            lines = page.extract_words()[::]  # 提取页面所有文本行  返回列表 包括：单词的文本内容单词在页面中的位置（坐标）
        except:
            return None   # 提取失败时返回 None
        if(len(lines) > 0):
            for line in lines:
                if("目录" in line["text"] or ".........." in line["text"]):  # 若某行文本包含“目录”或连续句点（常见于目录页码），直接返回 None。
                    return None   # 目录页不提取页眉
                if(line["top"] < 20 and line["top"] > 17):   # 优先选择页面顶部区域（top 在 17–20 之间）的文本作为页眉。
                    return line["text"]
            return lines[0]["text"]  # 未找到符合条件时返回首行文本
        return None

    '''
    ParseBlock 方法用于从 PDF 文档的每一页中提取结构化文本块，结合页眉（Header）进行组合，最终通过 Datafilter 方法过滤和存储数据。其核心逻辑如下：
        1.跳过目录页：通过 GetHeader 方法检测页眉，若为目录页则忽略。
        2.文本块分割：根据字体大小（size 属性）将页面内容划分为多个文本块。
        3.过滤特殊符号：忽略特定符号（如 □、•）和警告类文本（如“注意！”）。
        4.动态合并短文本：若相邻文本字体大小相同，则合并为同一块；若字体变化且当前块较短（长度 < 15），仍尝试合并。
        5.数据存储：将最终文本块与页眉组合后，调用 Datafilter 存储到 self.data。

    '''
    # 按照每页中块提取内容,并和一级标题进行组合,配合Document 可进行意图识别
    def ParseBlock(self, max_seq = 1024):

        with pdfplumber.open(self.pdf_path) as pdf: # 使用 pdfplumber 库打开 PDF 文件，生成 pdf 对象以逐页解析。

            for i, p in enumerate(pdf.pages):  # 逐页处理
                header = self.GetHeader(p)  # 调用 GetHeader 提取页眉

                if(header == None): # 若返回 None（是目录页），则跳过该页。
                    continue
                
                # use_text_flow=True：按文本流顺序提取，保留阅读顺序。
                # extra_attrs=["size"]：提取字体大小信息。
                texts = p.extract_words(use_text_flow=True, extra_attrs = ["size"])[::] # 

                squence = ""  # 当前文本块内容
                lastsize = 0  # 上一个文本块的字体大小

                for idx, line in enumerate(texts):  # 遍历页面文本行
                    if(idx <1): # 跳过索引为 0 的行（为页眉）。
                        continue
                    if(idx == 1):  # 跳过索引为 1 且为纯数字的行（可能是页码）。
                        # isdigit() 检查字符串中的所有字符是否都是数字字符
                        if(line["text"].isdigit()):  # 跳过索引为 1 且为纯数字的行（可能是页码）。 
                            continue
                    cursize = line["size"]
                    text = line["text"] 
                    if(text == "□" or text == "•"):  # 忽略复选框符号 □ 和项目符号 •。  
                        continue
                    elif(text== "警告！" or text == "注意！" or text == "说明！"):  # 遇到警告类文本（如“注意！”），保存当前块并重置 squence。
                        if(len(squence) > 0):
                            self.Datafilter(squence, header, i, max_seq = max_seq)
                        squence = ""
                    elif(format(lastsize,".5f") == format(cursize,".5f")):  # 字体大小相同  合并到当前块
                        # 合并到当前块
                        if(len(squence)>0):  
                            squence = squence + text  
                        else:
                            squence = text  
                    else:              # 字体大小不相同  
                        lastsize = cursize  # 保存该字体大小
                        if(len(squence) < 15 and len(squence)>0):  # 若当前块长度 < 15，继续合并（避免短文本孤立）
                            squence = squence + text
                        else:
                            if(len(squence) > 0):
                                self.Datafilter(squence, header, i, max_seq = max_seq)
                            squence = text
                if(len(squence) > 0):  # 处理循环结束后未保存的文本块。
                    self.Datafilter(squence, header, i, max_seq = max_seq)

    '''
    ParseOnePageWithRule 方法通过简单规则从 PDF 每页提取文本，并根据长度过滤、分割后存储到 self.data。核心逻辑如下：
        过滤无效内容：跳过目录页、空行、纯数字行。
        合并页面文本：将有效行拼接为完整页面内容。
        分割超长文本：若页面内容超过 max_seq，按句号（。）分割后合并成块。
        去重存储：确保不重复添加相同文本块。
    '''
    # 按句号划分文档，然后利用最大长度划分文档块
    def ParseOnePageWithRule(self, max_seq = 512, min_len = 6):
        # 遍历 PDF 每一页，提取原始文本并按换行符拆分成多行。
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")   # 按换行符分割为列表
            for idx, word in enumerate(words):   # 遍历每一行
                text = word.strip().strip("\n")   # 清理两端空白和换行符
                if("...................." in text or "目录" in text):
                    continue    # 跳过目录内容
                if(len(text) < 1):
                    continue    # 跳过空行
                if(text.isdigit()):     # 跳过纯数字行（如页码）
                    continue
                page_content = page_content + text  # 合并有效行
            if(len(page_content) < min_len):   # 若合并后的页面内容长度小于 min_len（6），则丢弃
                continue    # 跳过过短页面
            if(len(page_content) < max_seq):  
                if(page_content not in self.data):
                    self.data.append(page_content)   # 存储长度小于512的文本
            else:   
                sentences = page_content.split("。")  # 按句号分割（句号会被去除）
                cur = ""
                for idx, sentence in enumerate(sentences):
                    if(len(cur + sentence) > max_seq and (cur + sentence) not in self.data):
                        self.data.append(cur + sentence)  # 存储长块，合并上最后一个句子，长度刚好超过512
                        cur = sentence
                    else:
                        cur = cur + sentence  # 继续合并
 
    #  滑窗法提取段落
    #  1. 把pdf看做一个整体,作为一个字符串
    #  2. 利用句号当做分隔符,切分成一个数组
    #  3. 利用滑窗法对数组进行滑动, 此处的
    '''
    ParseAllPage 方法将整个 PDF 视为连续文本，通过以下步骤提取段落：
        合并所有页面内容：过滤无效行后，将有效文本拼接为单个字符串。
        按句号分割句子：将全文按中文句号 。 拆分为句子列表。
        滑窗生成段落：通过 SlidingWindow 方法将句子合并为不超过 max_seq 长度的段落。
    '''
    def ParseAllPage(self, max_seq = 512, min_len = 6):
        all_content = ""
        # 提取并过滤单页文本（逻辑与 ParseOnePageWithRule 相同）
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):   # 遍历每一页
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):    # 遍历每一页中的全部text内容
                text = word.strip().strip("\n")
                if("...................." in text or "目录" in text):
                    continue
                if(len(text) < 1):
                    continue
                if(text.isdigit()):
                    continue
                page_content = page_content + text
            if(len(page_content) < min_len): # 丢弃过短页面
                continue
            all_content = all_content + page_content   # 合并所有有效页面文本
        sentences = all_content.split("。")    # 按句号分割为列表（句号会被去除）
        self.SlidingWindow(sentences, kernel = max_seq)  #  将 PDF 视为整体文本，按句号分割后滑窗合并。


if __name__ == "__main__":
    # 主函数通过多种分块策略（按页、全局滑窗、按页规则）提取 PDF 文本，生成不同粒度的文本块，最终合并写入文件。适用于需要多维度分块分析的场景（如训练语言模型、信息检索）

    # 创建 DataProcess 类的实例 
    dp =  DataProcess(pdf_path = "./data/train_a.pdf")
    # 按照每页中块提取内容,将字体一样的文本内容放在一起，例如能够将（一级标题，警告标语）如：注意！ 警告！说明！这些内容块来进行存储。
    # 存储时若长度大于1024则按照可能存在的特殊符号（■，•,\t）进行分割)。
    dp.ParseBlock(max_seq = 1024)   
    dp.ParseBlock(max_seq = 512)
    print(len(dp.data))

    # 将 PDF 视为整体文本，将全部文本按句号分割为列表后，按照最大长度进行滑窗合并（步数是1）,存储到 self.data。
    dp.ParseAllPage(max_seq = 256)
    dp.ParseAllPage(max_seq = 512)
    print(len(dp.data))

    # 从 PDF 每页提取文本，并根据长度过滤、分割后存储到 self.data
    dp.ParseOnePageWithRule(max_seq = 256)
    dp.ParseOnePageWithRule(max_seq = 512)
    print(len(dp.data))

    # 将 dp.data 中所有文本块写入 all_text.txt 文件，每块占一行。
    data = dp.data
    out = open("all_text.txt", "w")
    for line in data:
        line = line.strip("\n")
        out.write(line)
        out.write("\n")
    out.close()
