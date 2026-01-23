import aiohttp  # 异步HTTP客户端/服务器库，用于高效发送网络请求
import asyncio  # 异步IO库，实现协程并发控制
import json  # JSON数据处理库，用于解析/生成JSON
import logging  # 日志记录库
import time  # 时间模块
from typing import List, Tuple
import numpy as np  


logger = logging.getLogger(__name__)   # 获取当前模块的日志记录器。

# 定义一个类型注解的列表 REQUEST_LATENCY，用于存储请求延迟的数据。
# List[Tuple[int, int, float]] 指定列表中元素的类型为元组，每个元组包含三个元素：两个整数 (int) 和一个浮点数 (float)。
REQUEST_LATENCY: List[Tuple[int, int, float]] = []  

API_URL = 'http://127.0.0.1:8000/v1/completions'  #  定义一个字符串 API_URL，存储 API 的 URL。
MODEL_UID = 'Qwen2_7B'   # 定义一个字符串 MODEL_UID，存储模型的唯一标识符。

# 定义一个字典 HEADERS，用于存储 HTTP 请求头信息。
HEADERS = {
    'Content-Type': 'application/json',   
}


'''
定义一个异步函数 send_request，用于发送 HTTP 请求
session  一个异步会话对象，用于发送 HTTP 请求
payload  请求的数据负载，通常是一个字典或 JSON 对象
prompt_len  提示的长度
'''

async def send_request(session, payload, prompt_len):
    # 记录请求开始的时间，用于计算请求延迟
    request_start_time = time.time()
    # 使用异步会话发送 POST 请求
    async with session.post(API_URL, data=payload, headers=HEADERS) as response:  # 请求的 URL。 请求的数据负载。 请求头信息
        # 检查响应的状态码是否为 200
        if response.status == 200: 
            result = await response.json()   # 解析响应数据为 JSON 格式
            completion_tokens = len(result['choices'][0]['text']) # 提取响应中完成的 token 数量
            # 记录请求结束的时间，并计算请求延迟
            request_end_time = time.time() 
            request_latency = request_end_time - request_start_time
            # 将请求延迟记录到 REQUEST_LATENCY 列表中
            REQUEST_LATENCY.append((prompt_len, completion_tokens, request_latency))
            return result
        else: # 如果响应状态码不是 200，返回一个包含错误信息的字典
            return {'error': response.status, 'message': await response.text()}


# 运行基准测试任务
class BenchMarkRunner:

    def __init__(
        self,
        requests: List[Tuple[str, int, int]],  # prompt, prompt_len, completion_len
        concurrency: int,
    ):
        self.concurrency = concurrency  # 设置并发任务的数量
        self.requests = requests    # 存储传入的请求列表
        self.request_left = len(requests)  # 记录剩余请求的数量
        self.request_queue = asyncio.Queue(concurrency or 100)  # 创建一个异步队列，用于存储请求，队列的大小为 concurrency 或 100，取两者中的较大值。
    
    # 定义一个异步方法
    async def run(self):
        # 创建并发任务列表 tasks
        tasks = []
        # 创建一个异步任务，并将这些任务添加到 tasks 列表
        for i in range(self.concurrency):
            tasks.append(asyncio.create_task(self.worker()))
        # 将所有请求添加到请求队列 self.request_queue 中
        for req in self.requests:
            await self.request_queue.put(req)
        # 等待所有任务完成。
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    #  异步函数  用于处理请求队列中的请求。
    async def worker(self):
        timeout = aiohttp.ClientTimeout(total=5 * 60)  # 如果请求在指定时间内未完成，则会抛出超时异常。 设置总超时时间为 5 分钟
        # 创建一个异步会话对象 session，用于发送 HTTP 请求。会话对象会在退出 with 块时自动关闭。
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while self.request_left > 0: # 表示还有请求需要处理
                # 从请求队列 self.request_queue 中获取一个请求  get() 方法会阻塞直到队列中有请求可用。
                prompt = await self.request_queue.get()
                # 构建请求负载 payload，其中包含模型 ID、提示文本和其他参数，然后将其转换为 JSON 格式的字符串。
                payload = json.dumps({"model": MODEL_UID,
                                      "prompt": f"Human: {prompt}\nAssistant: ",
                                      "stop": "<|endoftext|>",
                                      "temperature": 0,
                                      "max_tokens": 64,
                                      "top_k": 1
                                      })
                # 发送HTTP 请求并等待响应
                response = await send_request(session, payload, len(prompt))
                # 剩余请求数量减1
                self.request_left -= 1
                # 打印当前响应的索引
                print(f"Response {len(self.requests) - self.request_left}")


def main():
    concurrency = 50 # 设置并发任务的数量，即同时运行的任务数
    logger.info("Preparing for benchmark.") # 记录日志信息，表明基准测试准备开始。
    # 从 bench_data.json 文件加载测试数据集，并将其转换为输入请求列表
    testset = json.load(open("bench_data.json"))
    input_requests = list(testset.values()) 

    # 记录基准测试开始的时间。
    logger.info("Benchmark starts.")
    benchmark_start_time = time.time()

    # 开始基准测试 asyncio.run() 是用于运行异步程序的入口点
    asyncio.run(BenchMarkRunner(input_requests, concurrency).run())
    benchmark_end_time = time.time()
    # 记录基准测试结束的时间，并计算总耗时
    benchmark_time = benchmark_end_time - benchmark_start_time

    # 打印总耗时，
    print(f"Total time: {benchmark_time:.4f} s")
    # 计算并打印吞吐量，即每秒处理的请求数
    print(f"Throughput: {len(REQUEST_LATENCY) / benchmark_time:.2f} requests/s")

    # 计算所有请求的平均延迟，并打印结果。
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.4f} s")

    # 计算每个 token（输入和输出长度之和）的平均延迟，并打印结果
    avg_per_token_latency = np.mean(
        [
            latency / (prompt_len + output_len)
            for prompt_len, output_len, latency in REQUEST_LATENCY
        ]
    )

    print(f"Average latency per token: {avg_per_token_latency:.4f} s")

    # 计算每个输出 token 的平均延迟，并打印结果。
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY]
    )
    print("Average latency per output token: " f"{avg_per_output_token_latency:.4f} s")

    # 计算吞吐量，即每秒处理的 token 数，
    throughput = (
            sum([output_len for _, output_len, _ in REQUEST_LATENCY]) / benchmark_time
    )
    print(f"Throughput: {throughput} tokens/s")


if __name__ == '__main__':
    main()
