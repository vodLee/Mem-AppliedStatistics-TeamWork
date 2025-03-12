import json

# 读取原始数据
with open('./train_data_bad0.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 按id升序排序
sorted_data = sorted(data, key=lambda x: x['id'])

# 写回文件保持格式一致
with open('./train_data_bad0.json', 'w', encoding='utf-8') as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)

print("数据排序完成，已保存回原文件")