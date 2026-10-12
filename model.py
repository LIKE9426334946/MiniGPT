import math
from pathlib import Path

import torch
from torch import nn

from vocab import token_to_id


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model 必须能被正整数 num_heads 整除")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 三个独立的线性层，分别生成 Q、K、V。
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, padding_mask):
        B, T, C = x.shape

        # [B, T, 64]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 拆成 4 个 head：[B, T, 64] -> [B, 4, T, 16]
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # [B, 4, T, T]，最后两维分别对应 query 位置和 key 位置。
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # True 表示需要遮住的位置；上三角遮住未来的 token。
        causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # 同时遮住作为 key 的 <pad>：[B, T] -> [B, 1, 1, T]。
        scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))

        # 对每个 query，把它对各个 key 的分数变成注意力权重。
        attention_weights = torch.softmax(scores, dim=-1)

        # [B, 4, T, T] @ [B, 4, T, 16] -> [B, 4, T, 16]
        context = attention_weights @ v

        # 合并 4 个 head，恢复为 [B, T, 64]。
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, activation):
        super().__init__()
        activations = {"relu": nn.ReLU, "gelu": nn.GELU}
        if activation not in activations:
            raise ValueError("activation 目前支持 relu 或 gelu")

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)

        # 按当前配置：64 -> 192 -> 64。
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activations[activation](),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x, padding_mask):
        # Pre-LN：先归一化，再经过子层，最后加残差。
        x = x + self.attention(self.ln1(x), padding_mask)
        x = x + self.feed_forward(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config["model"]
        self.max_seq_len = cfg["max_seq_len"]
        self.pad_id = token_to_id["<pad>"]
        if cfg["vocab_size"] != len(token_to_id):
            raise ValueError("vocab_size 与词表大小不一致")

        # token ID 查表，得到可学习的 64 维词向量。
        self.token_embedding = nn.Embedding(
            cfg["vocab_size"], cfg["d_model"], padding_idx=self.pad_id
        )

        # 位置 0～31 也各有一个可学习的 64 维向量。
        self.position_embedding = nn.Embedding(cfg["max_seq_len"], cfg["d_model"])

        # 每次循环都创建新 Block，因此各层拥有独立参数。
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg["d_model"],
                    cfg["num_heads"],
                    cfg["dim_feedforward"],
                    cfg["activation"],
                )
                for _ in range(cfg["num_layers"])
            ]
        )

        self.final_norm = nn.LayerNorm(cfg["d_model"])
        self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"])

    def forward(self, token_ids):
        # token_ids: [B, T]，对应 dataset.py 返回的 x。
        if token_ids.ndim != 2:
            raise ValueError("token_ids 的形状必须是 [B, T]")
        B, T = token_ids.shape
        if not 1 <= T <= self.max_seq_len:
            raise ValueError(f"序列长度必须在 1～{self.max_seq_len} 之间")

        padding_mask = token_ids.eq(self.pad_id) # 不等于0的全部为False
        # 第一个位置必须有效，避免注意力的一整行都被遮住。
        # 上一步 dataset.py 已经在每条文本开头放了 <bos>。
        if padding_mask[:, 0].any():
            raise ValueError("输入应从有效 token 开始，并在右侧补 <pad>")

        positions = torch.arange(T, device=token_ids.device)
        token_vectors = self.token_embedding(token_ids)  # [B, T, 64]
        position_vectors = self.position_embedding(positions)  # [T, 64]
        x = token_vectors + position_vectors.unsqueeze(0)

        for block in self.blocks:
            x = block(x, padding_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # [B, T, 100]
        return logits


if __name__ == "__main__":
    import yaml
    from dataset import create_dataloaders

    config_path = Path(__file__).with_name("config.yaml")
    with config_path.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["train"]["seed"])
    model = MiniGPT(config)
    train_loader, _ = create_dataloaders(config_path)
    x, y = next(iter(train_loader))

    logits = model(x)
    print("输入形状：", x.shape)
    print("标签形状：", y.shape)
    print("输出形状：", logits.shape)
    print("参数总数：", sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss(ignore_index=token_to_id["<pad>"])
    loss = criterion(
        logits.reshape(-1, config["model"]["vocab_size"]),
        y.reshape(-1),
    )
    print("loss：", loss.item())

    # 检查梯度能否传回 Embedding；这里只计算梯度。
    loss.backward()
    print("Embedding 梯度形状：", model.token_embedding.weight.grad.shape)
