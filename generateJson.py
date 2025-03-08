import json

output_data = []
current_id = 0  # 初始化ID计数器

# 按行读取文件
with open('news2016zh_train.json', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()  # 去除首尾空白字符
        if not line:
            continue  # 跳过空行
        
        # 解析单行JSON
        item = json.loads(line)
        
        # 检查desc字段是否存在且非空
        if item.get("desc"):
            output_data.append({
                "id": current_id,
                "raw_data": item["content"]
            })
            current_id += 1  # ID递增

# 写入新文件
with open('tain_data.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
