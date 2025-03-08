import json

current_id = 0  # 初始化ID计数器

def truncate_text(text, max_length=2000):
    # 先进行基础裁剪
    truncated = text[:max_length]
    # 查找最后一个句子结束符（包括中文标点）
    last_sentence_end = max(
        truncated.rfind('。'),
        truncated.rfind('！'),
        truncated.rfind('？'),
        truncated.rfind('. '),  # 英文句号加空格
        truncated.rfind('! '),
        truncated.rfind('? ')
    )
    # 如果找到合适的结束位置，且距离末尾不超过50个字符
    # 50不够，还有很多句子超过50个字符，所以改成200
    # 为了保证完整性，大于200的咱不要了
    if last_sentence_end != -1 and (max_length - last_sentence_end) >= 200:
        return ""
    if last_sentence_end != -1 and (max_length - last_sentence_end) < 200:
        return truncated[:last_sentence_end+1]  # 包含结束符号
    return truncated

# 读取原始数据
with open('./train_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理并裁剪文本
processed_data = []
for item in data:
    content = item['raw_data']
    if len(content) > 2000:
        item['raw_data'] = truncate_text(content)
    # 字数太少的信息不足，咱不要了
    if len(item['raw_data']) > 500:
        item['id'] = current_id
        current_id += 1
        processed_data.append(item)

# 保存处理后的文件
with open('processed_train_data2.json', 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)
