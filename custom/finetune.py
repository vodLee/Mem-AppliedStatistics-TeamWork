# 导入所需模块和库，包含用于加载模型、配置低秩适应（LoRA）参数、定义数据预处理等功能
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    HfArgumentParser,
)
from peft import LoraConfig, TaskType
from arguments import ModelArguments, DataTrainingArguments, PeftArguments
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset
from data_preprocess import InputOutputDataset

def main():
    # 使用 HfArgumentParser 解析命令行参数，并将参数解析成数据类对象：model_args（模型相关）、data_args（数据相关）、peft_args（LoRA参数）、training_args（训练配置）
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments, RewardConfig))
    model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    # model = AutoModelForSequenceClassification.from_pretrained("gpt2")
    # 加载预训练的生成式语言模型 (AutoModelForCausalLM) 和分词器 (AutoTokenizer)。模型加载时设定了数据类型为 torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=peft_args.lora_rank, 
        lora_alpha=peft_args.lora_alpha, 
        lora_dropout=peft_args.lora_dropout
    )

    # 如果启用了 do_train 标志，读取训练数据文件（JSONL 格式）并加载为列表格式，然后通过 InputOutputDataset 类预处理数据
    if training_args.do_train:
        # with open(data_args.train_file, "r", encoding="utf-8") as f:
        #     train_data = json.load(f)
        # train_dataset = InputOutputDataset(train_data, tokenizer, data_args)
        train_data = load_dataset("json", data_files=data_args.train_file, split="train")

    # 如果启用了 do_eval 标志，类似地读取验证数据文件并加载为验证数据集
    if training_args.do_eval:
        eval_data = load_dataset("json", data_files=data_args.validation_file, split="train")
        # with open(data_args.validation_file, "r", encoding="utf-8") as f:
        #     eval_data = json.load(f)
        # eval_dataset = InputOutputDataset(eval_data, tokenizer, data_args)

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=eval_data if training_args.do_eval else None,
        peft_config=peft_config,
    )

    # 启用梯度检查点来降低显存使用，并开启输入梯度需求，以便更高效的梯度计算
    if training_args.do_train:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        trainer.train()

    # 如果启用评估标志，调用 trainer.evaluate() 执行评估
    if training_args.do_eval:
        trainer.evaluate()

if __name__ == "__main__":
    main()
