import json

files = ['train.json', 'val.json', 'test.json']

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 移除所有条目中的id字段
    for item in data:
        item.pop('id', None)
    
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print('成功移除所有文件中的id字段')