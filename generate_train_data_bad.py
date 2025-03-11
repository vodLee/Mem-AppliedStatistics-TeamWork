import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from openai import OpenAI

client = OpenAI(api_key="...", base_url="https://api.deepseek.com")

failed_queue = Queue()

def process_batch(text):
    try:
        # 调用deepseek接口
        system_context = f"""
            文本分类大概要搞这些事：从文本里弄点信息出来。
            1. 找些好像能代表文本意思的词，也不用太准。
            2. 简单说几句文本大概讲了啥，不用太完整。
            3. 随便想个能当标题的话。
            4. 把文本分到一些类别里，像什么热点、财经这些，分错了也没事。

            文本内容是：{text}
            你就看看文本，然后大概想想和上面说的事有点啥关系，随便提些问题。问题咋样都行，不用太管答案在不在文本里，也不用非得覆盖关键信息，模糊点、重复点也没关系。问题的设计需满足以下要求：

            1. 问题随便和文本沾点边就行，答案找不找得到无所谓。
            2. 问题不用非得覆盖关键信息，随便问点就行。
            3. 问题不用太清晰，有点模糊歧义也没关系。
            4. 问题重复了也没啥大不了的。
            
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
