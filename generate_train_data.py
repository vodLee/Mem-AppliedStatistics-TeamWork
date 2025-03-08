import json
from openai import OpenAI



field = f"""
    文本分类任务：需要从文本中提取以下信息：
    1. 文本关键词：能反映文本主要内容的关键词
    2. 文本摘要：几句话概述文本完整内容
    3. 标题：适合作为标题的文本
    4. 文本分类：将文本归类到预设的类别（如：热点、财经、科技、体育、娱乐、军事、国际、社会、法制、房产、汽车、教育、文化、旅游、历史、理论、生活、游戏、星座、时尚、情感、其他）
    """

# texts = f"""
#     英国石油天然气投资公司(UKOG)9日说，英格兰南部地底探测到规模巨大的油田，石油储量或许高达1000亿桶。这家公司的首席执行官史蒂芬·桑德森当天在接受英国媒体采访时表示，该公司认为英格兰南部地区地底储有巨量的石油资源，推测储量在500亿至1000亿桶之间，或许是英国过去30年来发现的最大陆上石油资源。这可是一个惊人的数字。要知道，苏格兰在北海的所有石油公司在过去40年的总产量也只有450亿桶。该公司预计，这些石油资源地下储存深度在2500英尺至3000英尺(762米至914米)之间，其中可以开发的比例为5%至15%，这意味着到2030年，这一地区的石油产量将能够满足英国10%至30%的石油需求。英格兰南部的石油开采活动已有数十年历史，开采地点分布在肯特郡、苏赛克斯郡、萨里郡和汉普郡等地。去年，英国地质勘探局曾发布报告称，英格兰南部地区拥有页岩油资源，储量约为22亿至85亿桶。这一爆炸性的公告发布后，UKOG的股票便开始疯涨。截至4月9日，UKOG的股价从原先的每股1.1便士一路狂飙至3.25便士，涨幅高达261.99%，最高价甚至冲到了4.7便士。股票市值瞬间飙升为原先的3倍，达到1822万英镑。
#     """

client = OpenAI(api_key="这个不给看", base_url="https://api.deepseek.com")

# response = client.chat.completions.create(
#     model="deepseek-reasoner",
#     messages=[
#         {"role": "user", "content": system_context},
#     ],
#     stream=False
# )

# print(response.choices[0].message.reasoning_content)
# print(response.choices[0].message.content)

# 新增处理函数
def process_batch(texts_list):
    import json
    
    results = []
    
    for text in texts_list:
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
        
        # 拼接reasoning和content
        combined = f"{response.choices[0].message.reasoning_content}\n{response.choices[0].message.content}"
        results.append(combined)
    
    return results

# 读取原始数据
with open('./page0_train_data.json', 'r') as f:
    original_data = json.load(f)[:10]  # 取前10条

# 提取raw_data列表
texts_list = [item["raw_data"] for item in original_data]

# 处理批量数据
cot_queries = process_batch(texts_list)

# 更新CoTQuery字段
for i in range(10):
    original_data[i]["CoTQuery"] = cot_queries[i]

# 写入新文件
with open('./train_data_demo.json', 'w') as f:
    json.dump(original_data, f, indent=2, ensure_ascii=False)
