import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from openai import OpenAI
import time

client = OpenAI(api_key="sk-6547400fc2204dd5b5dd82cd91197f96", base_url="https://api.deepseek.com")

failed_queue = Queue()

field = f"""
    文本分类任务：需要从文本中提取以下信息：
    1. 文本关键词：能反映文本主要内容的关键词
    2. 文本摘要：几句话概述文本完整内容
    3. 标题：适合作为标题的文本
    4. 文本分类：将文本归类到预设的类别
"""

def process_batch(data, max_retries=3):
    text = data["raw_data"]
    entry_id = data.get("id")
    
    for attempt in range(max_retries):
        try:
            system_context = f"""
                我现在有一个数据标注任务，描述如下：{field}。
                文本内容如下：{text}
                请根据文本内容提出尽可能多的问题，要求：
                1.问题必须基于文本内容且答案存在
                2.问题需覆盖文本关键信息
                3.问题表述清晰无歧义
                4.问题之间不应重复
                """
            
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "user", "content": system_context},
                ],
                stream=False
            )
            
            return {
                "id": entry_id,
                "CoTQuery": f"{response.choices[0].message.reasoning_content}\n{response.choices[0].message.content}"
            }
        except Exception as e:
            print(f"尝试 {attempt+1} 次失败: {e}")
            time.sleep(2 ** attempt)  # 指数退避
    
    failed_queue.put(data)
    return None

# 加载失败数据
with open('./failed_data.json', 'r') as f:
    failed_data = json.load(f)

# 加载原始数据建立ID映射
with open('./train_data_bad1.json', 'r') as f:
    original_data = json.load(f)
id_map = {item["id"]: item for item in original_data}

# 过滤有效失败条目（保留ID的）
valid_failed = [d for d in failed_data if "id" in d]

with ThreadPoolExecutor(max_workers=30) as executor:
    futures = {
        executor.submit(process_batch, data): data
        for data in valid_failed
    }
    
    for future in as_completed(futures):
        result = future.result()
        if result:
            # 精准替换原始数据中的条目
            original_entry = id_map.get(result["id"])
            if original_entry:
                original_entry["CoTQuery"] = result["CoTQuery"]

# 保存更新后的数据
with open('./train_data_bad1.json', 'w') as f:
    json.dump(original_data, f, indent=2, ensure_ascii=False)

# 保存新的失败数据
if not failed_queue.empty():
    new_failed = list(failed_queue.queue)
    with open('./failed_data_retry.json', 'w') as f:
        json.dump(new_failed, f, indent=2, ensure_ascii=False)