# 方法二：流式解析（适合超大文件）
import json
import ijson

count = 0
with open('tain_data.json', 'r', encoding='utf-8') as f:
    # 使用ijson流式解析器逐个读取数组项
    for item in ijson.items(f, 'item'):
        if len(item.get('raw_data', '')) > 2000:
            count += 1
        # 每处理10000条输出进度
        if count % 10000 == 0:
            print(f"已处理 {count} 条...")
print(f"最终结果：超过2000字符的条目数量为 {count}") 