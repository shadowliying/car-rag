import os  # 操作系统接口库  用于文件路径操作、环境变量管理（
import torch  # PyTorch深度学习框架
import time # 时间处理模块

from config import *  # 导入自定义配置文件
from vllm import LLM, SamplingParams   # 导入vLLM推理引擎   LLM: 大语言模型加速推理类  SamplingParams: 生成参数配置（温度/temperature、top_p等采样设置）
 
from transformers import AutoModelForCausalLM, AutoTokenizer  # Hugging Face Transformers库  自动加载因果语言模型   自动加载对应的文本分词器
from transformers import GenerationConfig   # 生成配置类  定义模型生成文本时的参数（最大长度、重复惩罚系数等）

# 通义千问工具函数
# - make_context: 构建对话上下文
# - decode_tokens: 特殊token解码处理
# - get_stop_words_ids: 获取停止词token ID（控制生成终止条件）
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids  

# 禁用tokenizer的并行模式（防止多线程冲突）
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设备配置
DEVICE = LLM_DEVICE  # 获取设备类型
DEVICE_ID = "0"   # 指定GPU设备ID
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合完整设备标识符（如"cuda:0"）

# 特殊文本标记常量
IMEND = "<|im_end|>"           # 对话结束标记
ENDOFTEXT = "<|endoftext|>"    # 通用文本结束标记

# 获取停止词token的id 
def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "raw":
        # 原始格式：当生成"Human:"和 将分词器的文档结束标识符（eod_id） 作为停止词 ID
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        # chatml格式：遇到im_end或im_start标记时停止
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        # 如果 chat_format 不是上述两种格式，则抛出 NotImplementedError 异常，提示未知的聊天格式。
        raise NotImplementedError(f"Unknown chat format {chat_format!r}") 
    return stop_words_ids

# 释放gpu显存
def torch_gc():
    if torch.cuda.is_available():   # 检查CUDA可用性
        with torch.cuda.device(CUDA_DEVICE):  # 指定设备上下文
            torch.cuda.empty_cache()  # 清空PyTorch的缓存分配器
            torch.cuda.ipc_collect()  # 回收IPC共享内存资源
'''
self.model = LLM(model=model_path,             
                tokenizer=model_path,      
                tensor_parallel_size=1, 
如果是多卡，可以自己把tensor_parallel_size并行度设置为卡数N  
'''
class ChatLLM(object):

    def __init__(self, model_path):
       # 初始化分词器
       self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        pad_token='<|extra_0|>', # 显式指定填充token
                        eos_token='<|endoftext|>', # 显式指定文本结束token
                        padding_side='left', # 左填充（适合生成式模型）
                        trust_remote_code=True  # 允许加载自定义模型代码
                    )
       
       # 加载生成配置
       self.generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=self.tokenizer.pad_token_id) # 保持与tokenizer的一致性
       self.tokenizer.eos_token_id = self.generation_config.eos_token_id  # 对齐分词器的eos_token_id与生成配置
       self.stop_words_ids = []  # 初始化停止词列表


       # 加载vLLM大模型 
       self.model = LLM(model=model_path,               # 模型路径
                            tokenizer=model_path,       # 复用模型路径的分词器
                            tensor_parallel_size=1,     # 如果是多卡，可以自己把并行度设置为卡数N  
                            trust_remote_code=True,     # 信任远程代码
                            gpu_memory_utilization=0.6, # 可以根据gpu的利用率自己调整这个比例
                            dtype="bfloat16")           # 使用bfloat16精度
       # 构建停止词ID集合
       for stop_id in get_stop_words_ids(self.generation_config.chat_format, self.tokenizer):
            self.stop_words_ids.extend(stop_id)   # 展开嵌套列表（如[[1],[2]]→[1,2]）
       self.stop_words_ids.extend([self.generation_config.eos_token_id])  # 追加EOS token

       # LLM的采样参数
       sampling_kwargs = {
            "stop_token_ids": self.stop_words_ids,  # 停止条件token集合
            "early_stopping": False,                # 不提前终止（生成完整长度）
            "top_p": 1.0,                           # 关闭核采样（top-p=1.0）
            "top_k": -1 if self.generation_config.top_k == 0 else self.generation_config.top_k, # 动态设置top-k
            "temperature": 0.0,                      # 零温度（贪婪解码）
            "max_tokens": 2000,                      # 最大生成长度限制
            "repetition_penalty": self.generation_config.repetition_penalty,       # 重复惩罚系数
            "n":1,                                   # 生成1个结果
            "best_of":2,                             # 从2个候选中选择最佳
            "use_beam_search":True                   # 启用束搜索（与temperature=0配合）
       }
       # 实例化采样参数对象
       self.sampling_params = SamplingParams(**sampling_kwargs)

    # 批量推理，输入一个batch，返回一个batch的答案
    def infer(self, prompts):
       # 步骤1：准备批量输入文本
       batch_text = []
       for q in prompts:  # 遍历每个提问
            # 构造符合格式要求的上下文
            # qwen_generation_utils.py中的方法  根据chat_format(raw,chatml)返回对应格式的模板
            # raw: raw_text = query    chatml: raw_text = \n{im_start}user\n{query}{im_end}\n{im_start}assistant\n
            raw_text, _ = make_context(    
              self.tokenizer,
              q,  # 当前问题
              system="You are a helpful assistant.",  # 系统提示语
              max_window_size=self.generation_config.max_window_size,  # 最大历史长度
              chat_format=self.generation_config.chat_format,  # 格式类型
            )
            batch_text.append(raw_text)  # 添加到批量输入列表
       # 步骤2：批量生成响应
       outputs = self.model.generate(batch_text, # 格式化后的输入列表
                                sampling_params = self.sampling_params  # 预配置的生成参数
                               )
       # 步骤3：后处理响应
       batch_response = []
       for output in outputs:  # 遍历每个生成的输出
           output_str = output.outputs[0].text   # 提取主要生成的文本
           # 去除结尾的特殊标记
           if IMEND in output_str:
               output_str = output_str[:-len(IMEND)]  # 截断标记
           if ENDOFTEXT in output_str:  # 检查通用结束标记
               output_str = output_str[:-len(ENDOFTEXT)]  # 截断标记
           batch_response.append(output_str)   # 存储清理后的结果

       # 步骤4：显存清理
       torch_gc()
       return batch_response  # 返回处理后的响应列表

if __name__ == "__main__":
    qwen7 = "/root/autodl-tmp/codes/pre_train_model/Qwen-7B-Chat"
    start = time.time()
    llm = ChatLLM(qwen7)
    test = ["吉利汽车座椅按摩","吉利汽车语音组手唤醒","自动驾驶功能介绍"]
    generated_text = llm.infer(test)
    print(generated_text)
    end = time.time()
    print("cost time: " + str((end-start)/60))
