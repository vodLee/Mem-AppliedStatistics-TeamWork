from torch.utils.data import Dataset

# 统一最大长度
max_length = 2048  # 根据你的硬件调整

# 定义一个名为 InputOutputDataset 的新类，它继承自 Dataset 类。这使得我们的数据集能够与 PyTorch 的数据加载器兼容，支持批处理和其他数据加载功能
class InputOutputDataset(Dataset):
    # 定义类的初始化方法，接收三个参数：data（数据集）、tokenizer（分词器）、args（其他参数配置）
    def __init__(self, data, tokenizer, args):
        # 调用父类 Dataset 的初始化方法，以确保数据集正确初始化。这是继承类的标准做法
        super(InputOutputDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length

    # 用于返回数据集的长度，满足 Python 的长度协议
    def __len__(self):
        return len(self.data)

    # 通过索引访问数据集中的样本
    def __getitem__(self, i):
        item = self.data[i]
        # add_special_tokens 不在开头加 special_tokens
        # 调用分词器，对构建的提示文本进行编码
        # 同时处理 chosen 和 rejected
        chosen = self.tokenizer(
            item["chosen"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        rejected = self.tokenizer(
            item["rejected"],
            max_length=self.max_target_length,  # 关键！保持相同长度
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # 返回一个字典，其中包含编码后的输入 ID、注意力掩码和标签，供模型训练使用
        return {
            "input_ids_chosen": chosen["input_ids"],
            "attention_mask_chosen": chosen["attention_mask"],
            "input_ids_rejected": rejected["input_ids"],
            "attention_mask_rejected": rejected["attention_mask"]
        }