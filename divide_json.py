import json
import os

# 创建输出目录
os.makedirs('page_train_data', exist_ok=True)

# 读取原始数据
with open('./processed_train_data2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 按3000条分页
chunk_size = 3000
for page_num, start_idx in enumerate(range(0, len(data), chunk_size)):
    end_idx = start_idx + chunk_size
    page_data = data[start_idx:end_idx]
    
    # 生成文件名
    filename = f'./page_train_data/page{page_num}_train_data.json'
    
    # 写入分页文件
    with open(filename, 'w', encoding='utf-8') as f_out:
        json.dump(page_data, f_out, ensure_ascii=False, indent=2)
