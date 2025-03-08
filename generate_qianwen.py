import os
import dashscope
import json
import time
import sys
import traceback

# 读取训练数据
with open('./page1_train_data.json', 'r') as f:
    original_data = json.load(f)[:10]

enhanced_data = []

try:
    with open('./prompt-python.md', 'r', encoding='utf-8') as file:
        content = file.read()
except FileNotFoundError:
    print("未找到 'prompt-python.md' 文件，请检查文件路径是否正确。")
except Exception as e:
    print(f"读取文件时发生错误: {e}")


for item in original_data:
    # item["raw_data"]
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            system_context = f"""
                我需要你作为一名专业的思维链与问题生成专家，协助我完成一项高质量的数据标注任务。

                任务描述：
                1. 从多样化的中文资讯文本{item["raw_data"]}（包括新闻报道、生活百科、个人经历分享、商业广告等）中提取核心信息与关键知识点，分析文本中的事实陈述、观点表达、因果关系和逻辑推理。需关注文本的主题意图、关键事实、数据对比、人物关系和背景知识，生成能够全面评估读者对文本理解深度与广度的多层次问题。
                2. 基于以上任务描述和以下文本内容，我需要你生成高质量的思维链(Chain of Thought, CoT)和查询问题(Query)，用于训练能同时评估问题质量及其推理逻辑的奖励模型。

                文本内容：{item["raw_data"]}

                {content}
            """
            response = dashscope.Generation.call(
                api_key="sk-9b7fee187bfb4d4c8a050363d5fc02a3",
                model="qwq-32b",
                messages=[{'role':'user','content':system_context}],
                stream=True
            )

            # 定义完整思考过程
            reasoning_content = ""
            # 定义完整回复
            answer_content = ""
            # 判断是否结束思考过程并开始回复
            is_answering = False

            print("=" * 20 + "思考过程" + "=" * 20)

            for chunk in response:
                # 严格空值检查
                if not chunk or not hasattr(chunk, 'output'):
                    continue
                if not chunk.output or not chunk.output.choices:
                    continue
                if len(chunk.output.choices) == 0:
                    continue
                
                msg = chunk.output.choices[0].message
                
                # 处理思考过程
                if msg.reasoning_content:
                    print(msg.reasoning_content, end="", flush=True)
                    reasoning_content += msg.reasoning_content
                # 处理正式回复
                elif msg.content:
                    if not is_answering:
                        print("\n" + "="*20 + "完整回复" + "="*20)
                        is_answering = True
                    print(msg.content, end="", flush=True)
                    answer_content += msg.content
            
            item["CoTQuery"] = answer_content
            enhanced_data.append(item)
            print(f"成功处理ID:{item['id']}")
            break
            
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb = traceback.extract_tb(exc_traceback)[-1]
            line_number = tb.lineno
            print(f"ID:{item['id']} 第{retry_count+1}次重试，错误发生在第{line_number}行: {str(e)}")
            retry_count +=1
            time.sleep(2)
    
    if retry_count == max_retries:
        print(f"ID:{item['id']} 处理失败，保留原始数据")
        enhanced_data.append(item)

# 写入文件
with open('./train_data_qianwen2.json', 'w') as f:
    json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

print(f"\n成功处理{len(enhanced_data)}条数据，已保存至train_data_qianwen0.json")