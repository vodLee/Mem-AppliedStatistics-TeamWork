# 方法一：直接加载整个JSON文件（适合较小文件）
import json

count = 0
total_length = 0  # 新增总字数统计变量
last_id = None    # 新增最后ID记录变量

# with open('./processed_train_data2.json', 'r', encoding='utf-8') as f:
with open('./page_train_data/page0_train_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for item in data:
        raw_data = item.get('raw_data', '')
        # 统计长度
        if len(raw_data) > 2000:
            count += 1
        # 累加总字数
        total_length += len(raw_data)
        # 记录最后ID
        last_id = item.get('id')
        
print(f"超过2000字符的条目数量：{count}")
print(f"raw_data字段总字数：{total_length}")  # 新增输出
print(f"最后一条数据的ID：{last_id}")        # 新增输出 

