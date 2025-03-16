# evaluate_reward.py
import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(base_model_path, peft_path):
    """加载基础模型和LoRA适配器"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, peft_path)
    model = model.merge_and_unload().to("cuda").eval()
    return tokenizer, model

def calculate_reward(model, tokenizer, text):
    """计算单条文本的奖励分数"""
    inputs = tokenizer(
        text,
        max_length=4096,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.item()

def analyze_scores(scores):
    """分数分布分析"""
    return {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "min": np.min(scores),
        "max": np.max(scores),
        "percentile_25": np.percentile(scores, 25),
        "percentile_75": np.percentile(scores, 75)
    }

def main(args):
    # 加载模型
    tokenizer, model = load_model(args.model, args.ckpt)
    
    # 初始化统计指标
    results = {
        "total_samples": 0,
        "correct": 0,
        "chosen_scores": [],
        "rejected_scores": [],
        "score_diffs": []
    }
    
    # 读取测试数据（JSON数组格式）
    with open(args.data, "r", encoding="utf-8") as f:
        test_data = json.load(f)  # 直接加载整个JSON数组
        
        for item in tqdm(test_data, desc="Processing samples"):
            if "chosen" not in item or "rejected" not in item:
                print(f"跳过无效样本: {item}")
                continue
                
            chosen = item["chosen"]
            rejected = item["rejected"]
            
            # 计算奖励分数
            chosen_score = calculate_reward(model, tokenizer, chosen)
            rejected_score = calculate_reward(model, tokenizer, rejected)
            
            # 更新统计
            results["total_samples"] += 1
            results["correct"] += int(chosen_score > rejected_score)
            results["chosen_scores"].append(chosen_score)
            results["rejected_scores"].append(rejected_score)
            results["score_diffs"].append(chosen_score - rejected_score)
    
    # 计算详细指标
    metrics = {
        "accuracy": results["correct"] / results["total_samples"] * 100,
        "score_analysis": {
            "chosen": analyze_scores(results["chosen_scores"]),
            "rejected": analyze_scores(results["rejected_scores"]),
            "diff": analyze_scores(results["score_diffs"])
        },
        "confusion_cases": []
    }
    
    # 找出典型错误样本（差异最小的3个和最大的3个）
    sorted_diffs = sorted(enumerate(results["score_diffs"]), key=lambda x: x[1])
    for idx, diff in sorted_diffs[:3] + sorted_diffs[-3:]:
        metrics["confusion_cases"].append({
            "index": idx,
            "diff_score": diff,
            "chosen_score": results["chosen_scores"][idx],
            "rejected_score": results["rejected_scores"][idx]
        })
    
    # 打印并保存结果
    print("\n===== 综合评估报告 =====")
    print(f"样本总数: {results['total_samples']}")
    print(f"偏好准确率: {metrics['accuracy']:.2f}%")
    print("\n分数分布分析:")
    print(json.dumps(metrics["score_analysis"], indent=2))
    
    # 保存结果时拼接完整文件路径
    output_file = os.path.join(args.output_dir, "eval_results.json")
    
    # 确保目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存至 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--ckpt", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--data", type=str, required=True, help="测试数据集路径（JSON数组格式）")
    parser.add_argument("--output_dir", type=str, required=True, help="测试结果路径")
    args = parser.parse_args()
    main(args)