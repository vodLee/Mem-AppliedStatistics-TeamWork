import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from openai import OpenAI

client = OpenAI(api_key="...", base_url="https://api.deepseek.com")

failed_queue = Queue()

field = f"""
    文本分类任务：需要从文本中提取以下信息：
    1. 文本关键词：能反映文本主要内容的关键词
    2. 文本摘要：几句话概述文本完整内容
    3. 标题：适合作为标题的文本
    4. 文本分类：将文本归类到预设的类别（如：热点、财经、科技、体育、娱乐、军事、国际、社会、法制、房产、汽车、教育、文化、旅游、历史、理论、生活、游戏、星座、时尚、情感、其他）
    """

def process_batch(text):
    try:
        # 调用deepseek接口
        system_context = f"""
            我现在有一个数据标注任务，描述如下：{field}。
            文本内容如下：{text}
            请你仔细读取文本内容，先熟悉了解文本的主要内容，再与我的标注任务描述中的任务联系起来得到我们需要的重要内容，并根据这些内容提出尽可能多的问题.问题的设计需满足以下要求：

            1.问题必须基于文本内容，且答案需在文本中明确存在。
            2.问题应覆盖文本中的关键信息，为文本块的核心内容。
            3.问题应清晰、简洁，避免模糊或歧义。
            4.每个问题需独立成句，且不与已提出的问题重复。

            请严格按照以下格式输出你提出的问题，问题为中文：
            1.Query: 问题1
            2.Query: 问题2
            ...
            """
        
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "user", "content": system_context},
            ],
            stream=False
        )
        return f"{response.choices[0].message.reasoning_content}\n{response.choices[0].message.content}"
    except Exception as e:
        failed_queue.put(text)
        return str(e)

with open('./page1_train_data.json', 'r') as f:
    original_data = json.load(f)[:1000]

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {
        executor.submit(process_batch, data["raw_data"]): data
        for data in original_data
    }
    
    for future in as_completed(futures):
        data = futures[future]
        try:
            data["CoTQuery"] = future.result()
            print(f"\n\n\n{data['id']}: {data['CoTQuery']}")
        except Exception as e:
            print(f"处理失败: {e}")

if not failed_queue.empty():
    failed_data = [{"raw_data": text} for text in list(failed_queue.queue)]
    with open('./failed_data.json', 'w') as f:
        json.dump(failed_data, f, indent=2, ensure_ascii=False)

with open('./train_data_bad0.json', 'w') as f:
    json.dump(original_data, f, indent=2, ensure_ascii=False)
