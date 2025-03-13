import json
import random
import argparse


def split_dataset(input_file, train_ratio=0.8, val_ratio=0.1):
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机打乱数据
    random.shuffle(data)
    total = len(data)
    
    # 计算分割点
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    
    # 分割数据集
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output_dir', default='.', help='Output directory')
    
    args = parser.parse_args()
    
    # 创建输出目录
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 执行分割
    train, val, test = split_dataset(args.input)
    
    # 保存结果文件
    with open(os.path.join(args.output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(args.output_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(args.output_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test, f, indent=2, ensure_ascii=False)
    
    print(f'Split completed: {len(train)} train, {len(val)} val, {len(test)} test samples')