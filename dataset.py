from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from vocab import id_to_token, token_to_id


class TextDataset(Dataset):
    def __init__(self, file_path, max_seq_len):
        if max_seq_len < 1:
            raise ValueError("max_seq_len 必须大于 0")

        pad_id = token_to_id["<pad>"]
        unk_id = token_to_id["<unk>"]
        bos_id = token_to_id["<bos>"]
        eos_id = token_to_id["<eos>"]
        self.samples = []

        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line_number, line in enumerate(f, start=1):
                # 当前语料的单词和标点已经用空格隔开。
                words = line.strip().lower().split()
                if not words:
                    continue

                # 单词转为 ID，并添加开始、结束标记。
                word_ids = [token_to_id.get(w, unk_id) for w in words]
                ids = [bos_id] + word_ids + [eos_id]

                # 每个输入位置的标签，都是它的下一个 token。
                x = ids[:-1]
                y = ids[1:]

                if len(x) > max_seq_len:
                    raise ValueError(
                        f"第 {line_number} 行需要 {len(x)} 个输入位置，"
                        f"超过 max_seq_len={max_seq_len}，请增大配置值。"
                    )

                # 在右侧补齐；x、y 的长度始终相同。
                pad_count = max_seq_len - len(x)
                x = x + [pad_id] * pad_count
                y = y + [pad_id] * pad_count
                self.samples.append((x, y))

        if not self.samples:
            raise ValueError("语料文件中没有有效文本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y = self.samples[index]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


def create_dataloaders(config_path=None):
    # 默认读取本文件旁边的 config.yaml。
    if config_path is None:
        config_path = Path(__file__).with_name("config.yaml")
    config_path = Path(config_path).resolve()

    with config_path.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)

    if config["model"]["vocab_size"] != len(token_to_id):
        raise ValueError("config.yaml 中的 vocab_size 与词表大小不一致")

    # 训练和验证分别读取各自的文件；相对路径以 config.yaml 所在目录为起点。
    train_path = (config_path.parent / config["data"]["train_path"]).resolve()
    val_path = (config_path.parent / config["data"]["val_path"]).resolve()
    if train_path == val_path:
        raise ValueError("训练集和验证集必须使用不同的语料文件")

    max_seq_len = config["model"]["max_seq_len"]
    train_set = TextDataset(train_path, max_seq_len)
    val_set = TextDataset(val_path, max_seq_len)
    seed = config["train"]["seed"]

    loader_args = {
        "batch_size": config["train"]["batch_size"],
        "num_workers": config["train"]["num_workers"],
    }

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_args,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **loader_args,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders()

    print("训练集数量：", len(train_loader.dataset))
    print("验证集数量：", len(val_loader.dataset))

    x, y = next(iter(train_loader))

    print("x.shape：", x.shape)
    print("y.shape：", y.shape)
    print("x.dtype：", x.dtype)
    print("y.dtype：", y.dtype)

    print("第一条输入 ID：", x[0].tolist())
    print("第一条标签 ID：", y[0].tolist())

    print(
        "第一条输入文本：",
        " ".join(id_to_token[i] for i in x[0].tolist()),
    )
    print(
        "第一条标签文本：",
        " ".join(id_to_token[i] for i in y[0].tolist()),
    )
