# PDF解析模块面试问答

## 一、基础理解类

### Q1: 请介绍一下你们项目中PDF解析模块的整体流程？

**答：** 我们的PDF解析模块主要分为以下几个步骤：

1. **PDF转图片**：将PDF每页转换为高清图片（300 DPI）
2. **版面分析**：使用PaddleOCR的版面分析功能，识别页面中的文档块、表格、图片等区域
3. **文本识别**：对文档块区域进行OCR文字识别
4. **表格提取**：对表格区域单独处理，保留表格结构信息
5. **数据清洗**：通过正则表达式过滤特殊字符、多余空格、无效内容
6. **文本分块**：将清洗后的文本按照滑动窗口策略切分成适合检索的chunks
7. **输出存储**：将处理后的文本块和表格数据保存，供后续向量化使用

---

### Q2: 为什么要对PDF进行分块（Chunking）？分块大小如何选择？

**答：** 分块的原因：
- **Embedding模型限制**：模型有最大输入长度限制（如512 tokens）
- **检索精度**：太大的文档块会稀释关键信息，导致检索不精确
- **上下文窗口**：LLM的上下文有限，需要控制召回内容长度
- **语义完整性**：合理分块能保持语义的完整性

分块大小选择：
- 我们实现了256、512、1024三种粒度
- **256字符**：细粒度，适合精确匹配
- **512字符**：平衡粒度，是主要使用的策略
- **1024字符**：粗粒度，保留更多上下文

实际选择需要根据：问题类型、文档特点、检索效果来调优。

---

### Q3: 你们实现了哪几种分块策略？各有什么优缺点？

**答：** 我们实现了三种分块策略：

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **ParseBlock** | 根据版面分析结果，按文档块边界分块 | 保持语义完整性，天然的段落边界 | 依赖版面分析准确性 |
| **ParseAllPage** | 全文拼接后用滑动窗口分块 | 简单通用，不依赖格式 | 可能切断语义 |
| **ParseOnePageWithRule** | 每页独立用固定长度分块 | 保持页面边界，适合表格类内容 | 跨页内容会被切断 |

实际使用中，我们根据文档特点选择合适的策略，汽车手册通常使用ParseBlock效果最好，因为PaddleOCR的版面分析能很好地识别章节结构。

---

## 二、技术细节类

### Q4: 滑动窗口分块的原理是什么？为什么要有重叠（overlap）？

**答：** 

**滑动窗口原理：**
```
文本: [A B C D E F G H I J K L M N O P]
窗口大小: 6, 步长: 4 (overlap=2)

第1块: [A B C D E F]
第2块: [E F G H I J]  ← 与第1块重叠2个
第3块: [I J K L M N]  ← 与第2块重叠2个
第4块: [M N O P]      ← 最后不足的部分
```

**为什么需要重叠（overlap）：**
1. **防止语义切断**：重要信息可能正好在切分边界，重叠确保信息在至少一个块中完整出现
2. **保持上下文连贯**：重叠部分提供前后文联系
3. **提高召回率**：同一信息在多个块中出现，增加被检索到的概率

我们的代码中使用20%的重叠率（chunk_size=512, overlap=102）。

---

### Q5: PDF解析用的什么工具？了解其他解析工具吗？

**答：** 我这边采用的是 **PaddleOCR** 按页来处理PDF，获得页面的所有文档块和表格。

我也调研过Python的其他PDF解析库：

| 工具 | 特点 | 优点 | 缺点 |
|------|------|------|------|
| **pdfplumber** | 基于PDFMiner和PyPDF2等底层库 | 提供更高层次的抽象和更友好的API，开发更容易上手 | 处理分栏的PDF或图文混排效果不好 |
| **pdfminer3k** | pdfminer的Python3版本 | 能提取PDF中的文本 | 表格处理非常不友好，能提取文字但没有格式 |
| **tabula / tabula-py** | 专门用来提取PDF表格数据 | 表格提取效果好 | 用Java写的，依赖Java8环境 |
| **camelot** | 用于PDF表格解析 | 能够处理复杂表格布局 | 也依赖Java8 |

**选择PaddleOCR的原因：**
1. 汽车用户手册包含扫描件页面，传统文本提取库无法处理
2. PaddleOCR支持版面分析（PP-Structure），能识别文档块、表格、图片等区域
3. 对中文识别效果好，百度开源的国产框架
4. **能同时处理文本和表格**，不需要额外引入Java依赖
5. 实际写代码时，可以几种工具结合着一起用

---

### Q6: 如何识别和处理目录页？

**答：** 我们通过以下方法识别目录页：

```python
def GetHeader(self, text):
    # 识别页眉模式
    header_pattern = r'([A-Za-z]+-\d+)'  # 如 "JMEV-01"
    match = re.search(header_pattern, text)
    return match.group(1) if match else None
```

**处理策略：**
1. 提取每页的页眉信息
2. 目录页通常没有特定的页眉格式，或者包含大量"....."连接符
3. 通过页眉模式匹配和内容特征来跳过目录页
4. 只保留正文页面的内容

---

### Q7: 数据清洗（Datafilter）具体做了哪些处理？

**答：** 我们的Datafilter方法做了以下清洗：

```python
def Datafilter(self, text):
    # 1. 移除多余空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 2. 移除特殊字符（保留中英文、数字、基本标点）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）\-\s]', '', text)
    
    # 3. 移除连续的标点
    text = re.sub(r'[，。！？]{2,}', '。', text)
    
    # 4. 去除首尾空格
    text = text.strip()
    
    return text
```

**清洗目的：**
- 减少噪音，提高文本质量
- 统一格式，便于后续处理
- 移除OCR识别产生的乱码和特殊字符

---

## 三、优化改进类

### Q8: 如果PDF中有表格，你们是怎么处理的？

**答：** 我们使用PaddleOCR的 **PP-Structure** 模块专门处理表格，这是我们项目的一个重点。

#### 1. 表格处理的完整流程

```
PDF页面 → 版面分析(识别表格区域) → 表格结构识别(行列划分) → 单元格OCR → 结构化输出(HTML) → 转Markdown → 建立索引
```

#### 2. 具体实现

```python
from paddleocr import PPStructure
from pdf2image import convert_from_path
import numpy as np

class TableExtractor:
    def __init__(self):
        # 初始化PP-Structure，开启表格识别
        self.engine = PPStructure(
            show_log=False,
            layout=True,      # 版面分析
            table=True,       # 表格结构识别
            ocr=True          # OCR识别
        )
    
    def extract_tables_from_pdf(self, pdf_path):
        """从PDF中提取所有表格"""
        images = convert_from_path(pdf_path, dpi=300)
        all_tables = []
        
        for page_num, img in enumerate(images):
            result = self.engine(np.array(img))
            
            for item in result:
                if item['type'] == 'table':
                    table_data = {
                        'page': page_num + 1,
                        'bbox': item['bbox'],           # 表格位置坐标
                        'html': item['res']['html'],    # HTML格式（保留结构）
                        'markdown': self.html_to_markdown(item['res']['html']),
                        'text': self.table_to_text(item['res']['html'])  # 纯文本
                    }
                    all_tables.append(table_data)
        
        return all_tables
```

#### 3. 表格转换格式

PP-Structure输出的是HTML格式的表格，我们会转换成多种格式：

**HTML格式（原始输出）：**
```html
<table>
  <tr><td>项目</td><td>规格</td><td>备注</td></tr>
  <tr><td>机油容量</td><td>4.5L</td><td>含滤清器</td></tr>
  <tr><td>冷却液容量</td><td>7.0L</td><td>-</td></tr>
</table>
```

**Markdown格式（便于检索）：**
```markdown
| 项目 | 规格 | 备注 |
|------|------|------|
| 机油容量 | 4.5L | 含滤清器 |
| 冷却液容量 | 7.0L | - |
```

**纯文本格式（用于向量化）：**
```
项目：机油容量，规格：4.5L，备注：含滤清器
项目：冷却液容量，规格：7.0L，备注：-
```

#### 4. 表格的索引策略

针对汽车手册中的表格（如参数表、保养周期表），我们采用**多粒度索引**：

```python
def index_table(self, table_data):
    """表格多粒度索引"""
    chunks = []
    
    # 策略1: 整表作为一个chunk（适合小表格）
    if len(table_data['text']) < 500:
        chunks.append({
            'content': table_data['markdown'],
            'type': 'table_full',
            'page': table_data['page']
        })
    
    # 策略2: 按行拆分（适合参数表、规格表）
    rows = self.parse_table_rows(table_data['html'])
    header = rows[0] if rows else []
    
    for row in rows[1:]:  # 跳过表头
        # 每行和表头组合，形成独立的检索单元
        row_text = self.format_row_with_header(header, row)
        chunks.append({
            'content': row_text,
            'type': 'table_row',
            'page': table_data['page']
        })
    
    return chunks

def format_row_with_header(self, header, row):
    """将表格行转换为自然语言描述"""
    # 例如: "机油容量的规格是4.5L，备注是含滤清器"
    parts = []
    for h, v in zip(header, row):
        if v and v != '-':
            parts.append(f"{h}：{v}")
    return "，".join(parts)
```

#### 5. 为什么要这样处理表格？

| 问题 | 解决方案 |
|------|----------|
| 表格转纯文本会丢失结构 | 保留HTML原始格式，同时生成Markdown |
| 整表太大不利于精确检索 | 按行拆分建立细粒度索引 |
| 单独一行缺少上下文 | 每行都带上表头信息 |
| 用户问"机油容量是多少" | 按行索引能精确召回"机油容量：4.5L" |

#### 6. 实际效果

汽车手册中有很多参数表，比如用户问"这款车的机油容量是多少"：
- 如果整表作为一个chunk，可能召回整个保养规格表（信息过多）
- 按行索引后，能精确召回"机油容量：4.5L，含滤清器"这一条

这种细粒度的表格处理，让我们在参数类问题上的准确率提升了约**15%**。

---

### Q9: 如何处理跨页的段落？

**答：** 这是PDF解析的经典难题。我们的处理策略：

**现有方案（ParseAllPage）：**
- 先将所有页面文本拼接
- 再用滑动窗口分块
- 自然处理了跨页问题

**改进思路：**
1. **段落延续检测**：如果页面末尾不是句号，可能是跨页段落
2. **版面分析辅助**：PaddleOCR能识别段落边界，跨页段落通常在下一页开头没有缩进
3. **语义相似度**：上一页末尾和下一页开头语义连贯则合并

```python
def merge_cross_page(page1_text, page2_text):
    # 如果page1不以句号结尾，与page2开头合并
    if not page1_text.rstrip().endswith(('。', '！', '？', '.', '!', '?')):
        return page1_text + page2_text
    return page1_text + '\n' + page2_text
```

---

### Q10: 分块质量如何评估？有什么指标？

**答：** 分块质量评估可以从以下维度：

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **语义完整性** | 每个chunk是否表达完整语义 | 人工标注 or LLM判断 |
| **检索命中率** | 相关chunk能否被召回 | Recall@K |
| **答案覆盖率** | 答案是否在召回的chunk中 | 标注数据验证 |
| **平均块长度** | 是否在合理范围 | 统计分析 |
| **块数量** | 文档被切分的块数 | 太多影响检索效率 |

**我们项目的评估：**
- 使用2000条测试集
- 召回Top10中命中率作为主要指标
- 最终系统准确率达到88.6%

---

## 四、场景应用类

### Q11: 汽车用户手册有什么特点？针对性做了哪些处理？

**答：** 汽车用户手册的特点：

1. **结构化强**：有明确的章节层级（大标题→小标题→正文）
2. **图文混排**：大量示意图配合文字说明
3. **表格较多**：参数配置、规格表、保养周期表等
4. **专业术语**：汽车相关的专有名词
5. **警告信息**：安全提示、注意事项格式特殊
6. **部分扫描件**：老版本手册可能是扫描的

**针对性处理：**
1. **使用PaddleOCR**：既能处理扫描件，又能识别版面结构
2. **表格专门处理**：使用PP-Structure提取表格，保留结构信息
3. **页眉识别**：通过特定格式（如JMEV-01）定位章节
4. **过滤无效页**：跳过目录、空白页、版权页
5. **保留警告标识**：如"注意"、"警告"等关键词优先保留

---

### Q12: 如果换成其他类型的文档（如论文、合同），需要修改什么？

**答：** 不同文档类型需要调整的内容：

| 文档类型 | 主要调整点 |
|----------|------------|
| **学术论文** | 1. 识别Abstract、Introduction等固定章节<br>2. 处理参考文献格式<br>3. 公式和图表编号处理 |
| **法律合同** | 1. 条款编号识别（第X条、X.X.X）<br>2. 保持条款完整性<br>3. 定义术语关联 |
| **技术文档** | 1. 代码块识别和保留格式<br>2. API文档结构化<br>3. 版本号处理 |

**核心思路：**
- 分析目标文档的结构特点
- 设计对应的章节识别规则
- 选择合适的分块策略
- 调整清洗规则保留关键信息
- PaddleOCR的版面分析对大多数文档类型都适用

---

## 五、代码细节类

### Q13: 请解释一下SlidingWindow函数的实现？

**答：** 

```python
def SlidingWindow(self, text, chunk_size=512, overlap=102):
    """
    滑动窗口分块
    
    参数:
        text: 待分块的文本
        chunk_size: 每个块的大小（字符数）
        overlap: 重叠区域大小
    
    返回:
        chunks: 分块后的文本列表
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        # 计算当前块的结束位置
        end = min(start + chunk_size, text_len)
        
        # 提取当前块
        chunk = text[start:end]
        chunks.append(chunk)
        
        # 滑动窗口：移动 (chunk_size - overlap) 的距离
        start += chunk_size - overlap
        
        # 如果已经到达末尾，退出
        if end == text_len:
            break
    
    return chunks
```

**关键点：**
- overlap参数控制重叠大小，一般设为chunk_size的20%
- 最后一个块可能小于chunk_size，这是正常的
- 步长 = chunk_size - overlap

---

### Q14: PaddleOCR的版面分析是怎么工作的？

**答：** 

PaddleOCR的 **PP-Structure** 模块提供版面分析功能：

```python
from paddleocr import PPStructure
from pdf2image import convert_from_path

class PDFParser:
    def __init__(self):
        # 初始化PP-Structure，开启版面分析和表格识别
        self.engine = PPStructure(
            show_log=False,
            layout=True,      # 开启版面分析
            table=True,       # 开启表格识别
            ocr=True          # 开启OCR
        )
    
    def parse_pdf(self, pdf_path):
        # 1. PDF转图片
        images = convert_from_path(pdf_path, dpi=300)
        
        all_blocks = []
        all_tables = []
        
        for page_num, img in enumerate(images):
            # 2. 版面分析 + OCR
            result = self.engine(np.array(img))
            
            for item in result:
                if item['type'] == 'text':
                    # 文本块
                    all_blocks.append({
                        'page': page_num,
                        'text': item['res']['text'],
                        'bbox': item['bbox']
                    })
                elif item['type'] == 'table':
                    # 表格
                    all_tables.append({
                        'page': page_num,
                        'html': item['res']['html'],
                        'bbox': item['bbox']
                    })
                elif item['type'] == 'title':
                    # 标题，可用于章节划分
                    all_blocks.append({
                        'page': page_num,
                        'text': item['res']['text'],
                        'is_title': True,
                        'bbox': item['bbox']
                    })
        
        return all_blocks, all_tables
```

**PP-Structure能识别的区域类型：**
- `text`: 普通文本段落
- `title`: 标题
- `table`: 表格
- `figure`: 图片
- `list`: 列表

---

### Q15: 如果OCR识别出来有错误，怎么处理？

**答：** 

**常见OCR错误类型：**

| 错误类型 | 表现 | 解决方案 |
|----------|------|----------|
| **相似字混淆** | "已"↔"己"，"0"↔"O" | 建立纠错词典，后处理替换 |
| **断字错误** | 一个字被拆成两部分 | 调高DPI，或使用更大的识别模型 |
| **多余字符** | 识别出背景噪点 | 图像预处理去噪，或过滤短字符串 |
| **漏识别** | 部分文字没有识别出 | 检查图片质量，调整置信度阈值 |

**代码层面的处理：**

```python
def post_process_ocr(self, text):
    # 1. 常见OCR错误纠正
    error_map = {
        '己经': '已经',
        '円': '圆',
        '|': 'I',  # 竖线误识别
    }
    for error, correct in error_map.items():
        text = text.replace(error, correct)
    
    # 2. 使用专业词典纠错（汽车术语）
    text = self.domain_correct(text)
    
    # 3. 过滤置信度低的结果
    # PaddleOCR返回每个字符的置信度，可以过滤低于阈值的
    
    return text

def domain_correct(self, text):
    # 汽车领域专业术语纠错
    car_terms = ['发动机', '变速箱', 'ESP', 'ABS', '安全气囊']
    # 使用编辑距离找相似词并纠正
    ...
```

**提升OCR准确率的方法：**
1. 提高图片DPI（我们用300）
2. 图像预处理：二值化、去噪、倾斜校正
3. 使用PaddleOCR的高精度模型
4. 针对特定领域微调OCR模型

---

## 六、改进提升类（高频追问）

### Q16: PDF中的图片你们是怎么处理的？

**答：** 汽车手册中有大量的示意图、操作图，这是一个重要但有挑战的问题。

**当前处理方案：**
```python
def extract_figures(self, page_result):
    """提取页面中的图片区域"""
    figures = []
    for item in page_result:
        if item['type'] == 'figure':
            figures.append({
                'bbox': item['bbox'],           # 图片位置
                'page': page_num,
                'nearby_text': self.get_nearby_text(item, page_result)  # 图片周围的文字
            })
    return figures

def get_nearby_text(self, figure_item, page_result):
    """获取图片周围的文字（图注、说明）"""
    # 通常图片下方或上方会有图注
    figure_bbox = figure_item['bbox']
    nearby = []
    for item in page_result:
        if item['type'] == 'text':
            # 判断文本块是否在图片附近
            if self.is_nearby(figure_bbox, item['bbox']):
                nearby.append(item['res']['text'])
    return ' '.join(nearby)
```

**我们的策略：**
1. **提取图注**：图片下方的说明文字，作为图片的文本描述
2. **上下文关联**：图片前后的段落文字，通常是对图片的解释
3. **建立关联索引**：`图片ID + 图注 + 上下文` 作为一个检索单元

**未来改进方向（多模态）：**

| 方案 | 说明 | 优缺点 |
|------|------|--------|
| **图片描述生成** | 用视觉模型（如BLIP-2）生成图片描述 | 能理解图片内容，但成本高 |
| **多模态Embedding** | 用CLIP等模型对图片和文本联合编码 | 支持跨模态检索，技术复杂 |
| **多模态LLM** | 用GPT-4V、Qwen-VL直接理解图片 | 效果最好，但推理成本高 |

```python
# 多模态改进示例（使用BLIP-2生成图片描述）
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def generate_image_caption(self, image):
    """用视觉模型生成图片描述"""
    inputs = self.blip_processor(images=image, return_tensors="pt")
    output = self.blip_model.generate(**inputs)
    caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

# 将生成的caption和图注一起建索引
figure_chunk = {
    'type': 'figure',
    'caption': figure_caption,           # 原始图注
    'generated_desc': generated_caption, # AI生成的描述
    'context': nearby_text               # 周围文字
}
```

---

### Q17: 如果让你重新做这个项目，你会怎么改进文档解析模块？

**答：** 我会从以下几个方面改进：

#### 1. 语义分块替代固定长度分块

```python
# 现有方案：固定512字符
chunks = sliding_window(text, chunk_size=512)

# 改进方案：基于语义边界分块
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", " "]  # 优先在句子边界切分
)
chunks = splitter.split_text(text)
```

#### 2. 引入父子文档索引（Parent Document Retriever）

```
问题：小chunk检索精准，但上下文不够；大chunk上下文够，但检索不精准

解决方案：同时建立两层索引
- 子文档（小chunk）：用于检索，更精准
- 父文档（大chunk）：检索到子文档后，返回其父文档，上下文更完整
```

```python
class ParentDocumentRetriever:
    def __init__(self):
        self.child_chunks = []   # 小块，用于检索
        self.parent_chunks = []  # 大块，用于返回
        self.child_to_parent = {}  # 子块到父块的映射
    
    def build_index(self, text):
        # 先切大块（1024字符）
        parent_chunks = sliding_window(text, chunk_size=1024, overlap=100)
        
        for i, parent in enumerate(parent_chunks):
            # 大块再切小块（256字符）
            children = sliding_window(parent, chunk_size=256, overlap=50)
            for child in children:
                child_id = len(self.child_chunks)
                self.child_chunks.append(child)
                self.child_to_parent[child_id] = i
        
        self.parent_chunks = parent_chunks
    
    def retrieve(self, query, top_k=3):
        # 用小块检索
        child_results = self.search_children(query, top_k)
        
        # 返回对应的大块
        parent_ids = set(self.child_to_parent[c['id']] for c in child_results)
        return [self.parent_chunks[pid] for pid in parent_ids]
```

#### 3. 元数据增强

```python
# 现有：chunk只有文本内容
chunk = "机油容量为4.5L，含滤清器"

# 改进：chunk携带丰富的元数据
chunk = {
    'content': "机油容量为4.5L，含滤清器",
    'metadata': {
        'source': '用户手册.pdf',
        'page': 45,
        'chapter': '保养规格',
        'section': '发动机保养',
        'type': 'table_row',        # chunk类型
        'keywords': ['机油', '容量', '保养'],  # 关键词
        'importance': 0.8           # 重要性评分
    }
}
```

#### 4. 分块质量自动评估

```python
def evaluate_chunk_quality(self, chunk):
    """自动评估分块质量"""
    scores = {}
    
    # 1. 长度合理性
    scores['length'] = 1.0 if 200 < len(chunk) < 800 else 0.5
    
    # 2. 语义完整性（是否以完整句子结尾）
    scores['completeness'] = 1.0 if chunk.rstrip().endswith(('。', '！', '？')) else 0.7
    
    # 3. 信息密度（非停用词比例）
    scores['density'] = self.calc_info_density(chunk)
    
    # 4. 用LLM判断语义完整性
    scores['semantic'] = self.llm_judge_completeness(chunk)
    
    return sum(scores.values()) / len(scores)
```

---

### Q18: 这个模块遇到过什么困难？怎么解决的？

**答：** 

| 困难 | 具体表现 | 解决方案 |
|------|----------|----------|
| **扫描件质量差** | 部分老手册是扫描的，OCR识别率低 | 1. 图像预处理（去噪、增强对比度）<br>2. 提高DPI到300<br>3. 多次识别取置信度最高的 |
| **表格跨页** | 大表格被分到两页，结构断裂 | 1. 检测表格是否在页面底部被截断<br>2. 下一页开头如果还是表格，尝试合并 |
| **分栏PDF** | 两栏排版，阅读顺序错乱 | PP-Structure的版面分析能自动识别分栏，按正确顺序输出 |
| **图注分离** | 图片和说明文字被分开 | 基于位置关系，将图片下方/上方的文字关联到图片 |
| **特殊符号** | 汽车手册有很多特殊符号（警告⚠️等） | 建立符号到文字的映射表，如 ⚠️ → [警告] |

**一个具体案例：**

> 问题：有些PDF页面是横向的（landscape），OCR识别出来的文字顺序全乱了
> 
> 排查：发现PaddleOCR默认假设页面是纵向的
> 
> 解决：
> ```python
> def detect_orientation(self, image):
>     """检测页面方向"""
>     h, w = image.shape[:2]
>     if w > h * 1.2:  # 宽度明显大于高度
>         # 横向页面，旋转90度
>         image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
>     return image
> ```

---

### Q19: 分块大小512是怎么确定的？试过其他值吗？

**答：** 512不是随便定的，是通过实验确定的。

**实验过程：**

| chunk_size | overlap | 召回率@10 | 平均chunk数 | 观察 |
|------------|---------|-----------|-------------|------|
| 256 | 50 | 89.2% | 3200 | 召回高但chunk太多，检索慢 |
| 512 | 100 | 87.5% | 1600 | **平衡点** |
| 1024 | 200 | 82.3% | 800 | chunk太大，精确度下降 |
| 动态 | - | 88.1% | 1400 | 基于句子边界，效果好但实现复杂 |

**选择512的原因：**
1. **Embedding模型**：M3E的最佳输入长度是512 tokens，中文约512字符
2. **检索效率**：chunk数量适中，检索速度可接受
3. **语义完整性**：512字符大约是2-3个完整段落，语义相对完整
4. **LLM上下文**：召回Top5约2500字符，不会超过LLM上下文

**20%重叠率的选择：**
- 太小（10%）：重要信息可能被切断
- 太大（30%）：冗余太多，chunk数量膨胀
- 20%是业界常用的经验值

---

### Q20: 如果文档量很大（比如1000个PDF），解析效率怎么优化？

**答：** 

**优化策略：**

```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

class BatchPDFParser:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.engine = None  # 每个进程单独初始化
    
    def parse_single_pdf(self, pdf_path):
        """解析单个PDF（在子进程中执行）"""
        if self.engine is None:
            # 懒加载，每个进程初始化一次
            from paddleocr import PPStructure
            self.engine = PPStructure(show_log=False)
        
        # 解析逻辑...
        return result
    
    def parse_batch(self, pdf_paths):
        """批量解析PDF"""
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self.parse_single_pdf, pdf_paths))
        return results
```

**效率对比：**

| 方案 | 1000个PDF耗时 | 说明 |
|------|---------------|------|
| 单进程串行 | ~50小时 | 基准 |
| 4进程并行 | ~13小时 | 线性加速 |
| 8进程并行 | ~7小时 | 受CPU核数限制 |
| GPU加速OCR | ~3小时 | PaddleOCR支持GPU |
| 增量更新 | - | 只解析新增/修改的PDF |

**其他优化：**
1. **缓存机制**：解析结果缓存，避免重复解析
2. **增量更新**：基于文件hash判断是否需要重新解析
3. **分布式处理**：多机并行，适合超大规模
4. **GPU加速**：PaddleOCR支持GPU推理，速度提升5-10倍

---

## 总结

PDF解析是RAG系统的基础模块，面试重点关注：
1. **工具选型**：为什么选择PaddleOCR，了解其他工具的优缺点
2. **分块策略的理解**：为什么分块、如何分块、如何评估
3. **表格处理**：如何提取和索引表格数据（PP-Structure）
4. **图片处理**：当前方案（图注+上下文）和未来改进（多模态）
5. **问题处理**：跨页、OCR错误、乱码等常见问题
6. **优化思路**：语义分块、父子文档、元数据增强
7. **工程能力**：大规模文档的并行处理、效率优化

建议结合项目实际数据和效果来回答，展示对细节的掌握。
