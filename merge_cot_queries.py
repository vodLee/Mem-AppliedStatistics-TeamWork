import json

with open('./train_data_good0.json', 'r') as f:
    good_data = json.load(f)

with open('./train_data_bad0.json', 'r') as f:
    bad_data = json.load(f)

assert len(good_data) == len(bad_data), "文件条目数不一致"

result = [
    {
        "id": idx,
        "chosen": good["CoTQuery"],
        "rejected": bad["CoTQuery"]
    }
    for idx, (good, bad) in enumerate(zip(good_data, bad_data))
]

with open('merged_cot_queries.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"已生成{len(result)}条记录")