import torch.nn as nn

class RewardModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pooler = nn.AdaptiveAvgPool1d(1)  # 序列维度聚合

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, seq_len]
        pooled = self.pooler(logits.unsqueeze(1)).squeeze(-1)  # [batch_size]
        return pooled