import argparse
import csv
import random
from pathlib import Path

import torch
import yaml
from torch import nn

from dataset import create_dataloaders
from model import MiniGPT
from vocab import token_to_id, tokens


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA，请将 device 改为 cpu 或 auto")
    return device


def run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=0.0):
    # 传入 optimizer 就训练；不传入时就验证。
    is_training = optimizer is not None
    model.train(is_training)
    pad_id = token_to_id["<pad>"]
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for x, y in loader:
        x = x.to(device)  # [B, T]
        y = y.to(device)  # [B, T]

        if is_training:
            # 1. 清空上一个 batch 的梯度。
            optimizer.zero_grad(set_to_none=True)

        # 验证时关闭梯度记录。
        with torch.set_grad_enabled(is_training):
            # 2. 前向传播，计算每个位置对各个 token 的预测分数。
            logits = model(x)  # [B, T, vocab_size]

            # 3. 计算损失。直接使用 logits，无须先做 Softmax。
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("loss 出现 NaN 或 Inf，请检查输入和学习率")

            if is_training:
                # 4. 反向传播：计算所有可学习参数的梯度。
                loss.backward()

                # 可选：限制梯度范数，减小异常大梯度的影响。
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                # 5. 根据梯度更新参数。
                optimizer.step()

        # 只统计非 <pad> 的标签位置。
        valid_mask = y.ne(pad_id)
        valid_count = valid_mask.sum().item()
        predictions = logits.detach().argmax(dim=-1)
        correct_count = ((predictions == y) & valid_mask).sum().item()

        # criterion 返回有效 token 的平均损失。
        # 按有效 token 数加权，避免不同 batch 长度造成统计偏差。
        total_loss += loss.item() * valid_count
        total_correct += correct_count
        total_tokens += valid_count

    if total_tokens == 0:
        raise ValueError("没有可用于计算损失的有效 token")
    return {
        "loss": total_loss / total_tokens,
        "accuracy": total_correct / total_tokens,
    }


def main(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    train_cfg = config["train"]
    if train_cfg["epochs"] < 1:
        raise ValueError("epochs 必须至少为 1")

    set_seed(train_cfg["seed"])
    device = get_device(train_cfg["device"])
    model = MiniGPT(config).to(device)
    train_loader, val_loader = create_dataloaders(config_path)

    criterion = nn.CrossEntropyLoss(ignore_index=token_to_id["<pad>"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    output_dir = config_path.parent / train_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history_path = output_dir / "history.csv"
    best_val_loss = float("inf")

    print(f"设备：{device}")
    print(f"训练集：{len(train_loader.dataset)} 条，验证集：{len(val_loader.dataset)} 条")
    print(f"模型参数：{sum(p.numel() for p in model.parameters()):,}")

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
        )
        writer.writeheader()

        for epoch in range(1, train_cfg["epochs"] + 1):
            train_metrics = run_epoch(
                model, train_loader, criterion, device,
                optimizer=optimizer, grad_clip=train_cfg["grad_clip"],
            )
            val_metrics = run_epoch(model, val_loader, criterion, device)

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_acc": val_metrics["accuracy"],
            }
            writer.writerow(row)
            f.flush()

            # 依据验证集 loss 选择最佳模型。
            is_best = val_metrics["loss"] < best_val_loss
            best_val_loss = min(best_val_loss, val_metrics["loss"])
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "vocab": list(tokens),
                # 使用独立语料文件后，Dataset 不再具有 random_split 的 indices。
                "train_path": config["data"]["train_path"],
                "val_path": config["data"]["val_path"],
                "train_size": len(train_loader.dataset),
                "val_size": len(val_loader.dataset),
                "best_val_loss": best_val_loss,
                **row,
            }
            torch.save(checkpoint, last_path)
            if is_best:
                torch.save(checkpoint, best_path)

            print(
                f"Epoch {epoch:02d}/{train_cfg['epochs']:02d} | "
                f"train_loss={row['train_loss']:.4f} | "
                f"val_loss={row['val_loss']:.4f} | "
                f"train_acc={row['train_acc']:.2%} | "
                f"val_acc={row['val_acc']:.2%}"
                + (" | 已保存最佳模型" if is_best else ""),
                flush=True,
            )

    print(f"训练结束，最佳验证 loss：{best_val_loss:.4f}")
    print(f"最佳模型：{best_path}")
    print(f"最后一轮模型：{last_path}")
    print(f"训练记录：{history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 MiniGPT")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="配置文件路径，默认读取 train.py 旁边的 config.yaml",
    )
    args = parser.parse_args()
    main(args.config)

