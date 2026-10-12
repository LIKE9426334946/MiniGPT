"""查看 MiniGPT 训练后的输入词向量；只读取 checkpoint，不修改模型。

运行：
    python inspect_embeddings.py
    python inspect_embeddings.py --token cat
    python inspect_embeddings.py --compare cat dog
    python inspect_embeddings.py --neighbors cat --top-k 10
    python inspect_embeddings.py --matrix cat dog i you

默认加载同目录 outputs/best.pt，也可使用 --checkpoint 指定路径。
只依赖 PyTorch；词表和 embedding 都来自上一版 train.py 保存的 checkpoint。
"""

import argparse
import math
from pathlib import Path

import torch

SPECIAL_TOKENS = {"<pad>", "<unk>", "<bos>", "<eos>"}
METRICS = ("cosine", "l2", "l1", "dot")


def load_embeddings(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"找不到模型文件：{checkpoint_path}\n"
            "请把脚本放在项目根目录，或用 --checkpoint 指定 best.pt。"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("请使用上一版 train.py 保存的 checkpoint")
    vocab = checkpoint.get("vocab")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(vocab, (list, tuple)) or not vocab:
        raise ValueError("checkpoint 中没有有效的 vocab 词表")
    if not all(isinstance(token, str) for token in vocab) or len(set(vocab)) != len(
        vocab
    ):
        raise ValueError("checkpoint 中的词表包含重复项或非字符串项")
    if not isinstance(state_dict, dict) or "token_embedding.weight" not in state_dict:
        raise ValueError(
            "checkpoint 中找不到 model_state_dict['token_embedding.weight']"
        )

    # 与 model.token_embedding.weight 是同一张训练后的参数表。
    # 第 0 维是 token ID，第 1 维是向量的各个分量：[100, 64]。
    weights = state_dict["token_embedding.weight"]
    if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
        raise ValueError("token_embedding.weight 应该是二维 tensor")
    if weights.shape[0] != len(vocab) or weights.shape[1] < 1:
        raise ValueError("embedding 的形状与词表不匹配")
    if not weights.is_floating_point() or not torch.isfinite(weights).all():
        raise ValueError("embedding 必须是有限的浮点数，不能包含 NaN 或 Inf")

    # 仅在内存中转为 float64，减小相似度/夹角计算的舍入误差。
    embeddings = weights.detach().to(device="cpu", dtype=torch.float64)
    print(f"已加载：{checkpoint_path}")
    print(f"词表大小：{len(vocab)}，每个 token 的维度：{embeddings.shape[1]}")
    print("查看对象：训练后的输入 token embedding（加入位置编码之前）")
    return list(vocab), embeddings


def resolve_token(text, vocab):
    """可输入 token 本身，也可输入它的十进制 ID。"""
    text = text.strip()
    if text in vocab:
        return vocab.index(text)
    if text.lower() in vocab:
        return vocab.index(text.lower())
    try:
        token_id = int(text)
    except ValueError:
        raise ValueError(
            f"词表中没有 {text!r}；可通过菜单 5 或 --vocab 查看词表"
        ) from None
    if not 0 <= token_id < len(vocab):
        raise ValueError(f"token ID 应在 0～{len(vocab) - 1} 之间")
    return token_id


def vector_metrics(a, b):
    """两个一维向量的指标；所有计算都针对完整向量，不做降维。"""
    difference = a - b
    l2 = torch.linalg.vector_norm(difference, ord=2).item()
    l1 = difference.abs().sum().item()
    dot = torch.dot(a, b).item()
    norm_a = torch.linalg.vector_norm(a).item()
    norm_b = torch.linalg.vector_norm(b).item()

    # cos(a, b) = (a · b) / (||a|| * ||b||)。
    # 零向量没有方向，不能把它与其他向量的余弦当作正常的 0。
    if norm_a == 0.0 or norm_b == 0.0:
        cosine = cosine_distance = angle = float("nan")
    else:
        cosine = torch.dot(a / norm_a, b / norm_b).clamp(-1.0, 1.0).item()
        cosine_distance = 1.0 - cosine
        angle = math.degrees(math.acos(cosine))

    return {
        "l2": l2,
        "l1": l1,
        "dot": dot,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "cosine": cosine,
        "cosine_distance": cosine_distance,
        "angle": angle,
    }


def format_value(value):
    return "未定义" if math.isnan(value) else f"{value:.6f}"


def show_vector(vocab, embeddings, token):
    token_id = resolve_token(token, vocab)
    vector = embeddings[token_id]  # [64]，这一行取出指定 token 的向量。
    print(f"\ntoken：{vocab[token_id]} | ID：{token_id} | 形状：{tuple(vector.shape)}")
    print("完整向量（每项格式为 维度索引:数值，索引从 0 开始）：")
    for start in range(0, vector.numel(), 8):
        end = min(start + 8, vector.numel())
        print(
            "  "
            + "  ".join(f"{i:02d}:{vector[i].item():+.6f}" for i in range(start, end))
        )
    norm = torch.linalg.vector_norm(vector).item()
    print(f"L2 范数（向量长度）：{norm:.6f}")
    if norm == 0.0:
        print("这是零向量，与任何向量的余弦相似度和夹角都未定义。")


def show_comparison(vocab, embeddings, first, second):
    first_id = resolve_token(first, vocab)
    second_id = resolve_token(second, vocab)
    a, b = embeddings[first_id], embeddings[second_id]
    values = vector_metrics(a, b)
    print(
        f"\n比较：{vocab[first_id]} (ID={first_id}) 与 {vocab[second_id]} (ID={second_id})"
    )
    labels = (
        ("norm_a", "第一个向量的 L2 范数"),
        ("norm_b", "第二个向量的 L2 范数"),
        ("l2", "欧氏距离 L2，越小越接近"),
        ("l1", "曼哈顿距离 L1，越小越接近"),
        ("dot", "点积，同时受长度和方向影响"),
        ("cosine", "余弦相似度，越接近 1 方向越一致"),
        ("cosine_distance", "余弦距离 1-cos，越小方向越接近"),
        ("angle", "向量夹角（度）"),
    )
    for key, label in labels:
        print(f"{label}：{format_value(values[key])}")
    if math.isnan(values["cosine"]):
        print("至少一个向量为零向量，因此余弦相似度、余弦距离和夹角未定义。")
    return values


def find_neighbors(
    vocab, embeddings, token, top_k=10, metric="cosine", include_special=False
):
    if top_k < 1:
        raise ValueError("top_k 必须至少为 1")
    if metric not in METRICS:
        raise ValueError("排序指标请选择 cosine、l2、l1 或 dot")
    token_id = resolve_token(token, vocab)
    query = embeddings[token_id]
    if metric == "cosine" and torch.linalg.vector_norm(query).item() == 0.0:
        raise ValueError("零向量无法按余弦相似度排序，可以改用 l2 或 l1")

    rows = []
    for other_id, other_token in enumerate(vocab):
        # 默认不把自己和特殊 token 放进候选列表。
        if other_id == token_id or (
            not include_special and other_token in SPECIAL_TOKENS
        ):
            continue
        values = vector_metrics(query, embeddings[other_id])
        if math.isnan(values[metric]):
            continue
        rows.append((other_id, values))
    descending = metric in {"cosine", "dot"}
    rows.sort(key=lambda row: row[1][metric], reverse=descending)
    return token_id, rows[:top_k]


def show_neighbors(
    vocab, embeddings, token, top_k=10, metric="cosine", include_special=False
):
    token_id, rows = find_neighbors(
        vocab, embeddings, token, top_k, metric, include_special
    )
    order = "从大到小" if metric in {"cosine", "dot"} else "从小到大"
    print(
        f"\n查询 token：{vocab[token_id]} | 按 {metric} {order}排列，返回 {len(rows)} 项"
    )
    print(
        "已排除自身；"
        + ("候选包含特殊 token。" if include_special else "候选排除特殊 token。")
    )
    if metric == "cosine":
        print("零向量不参与余弦排序。")
    print(
        f"{'排名':>4} {'ID':>4} {'token':>12} {'cosine':>12} {'L2':>12} {'L1':>12} {'dot':>12}"
    )
    for rank, (other_id, values) in enumerate(rows, 1):
        columns = " ".join(
            f"{format_value(values[key]):>12}" for key in ("cosine", "l2", "l1", "dot")
        )
        print(f"{rank:>4} {other_id:>4} {vocab[other_id]:>12} {columns}")


def show_matrices(vocab, embeddings, selected):
    if len(selected) < 2:
        raise ValueError("请至少输入两个 token，用空格隔开")
    ids = [resolve_token(token, vocab) for token in selected]
    names = [vocab[token_id] for token_id in ids]
    width = max(12, max(len(name) for name in names) + 2)
    values = [[vector_metrics(embeddings[i], embeddings[j]) for j in ids] for i in ids]
    for key, title in (("cosine", "余弦相似度矩阵"), ("l2", "欧氏距离 L2 矩阵")):
        print(f"\n{title}（行 token 与列 token 的比较）：")
        print(" " * width + "".join(f"{name:>{width}}" for name in names))
        for name, row in zip(names, values):
            print(
                f"{name:>{width}}"
                + "".join(f"{format_value(cell[key]):>{width}}" for cell in row)
            )
    print("未定义表示该比较包含零向量。")


def show_vocab(vocab):
    print("\n完整词表（ID:token）：")
    for start in range(0, len(vocab), 5):
        print(
            "  ".join(
                f"{i:>3}:{vocab[i]:<12}"
                for i in range(start, min(start + 5, len(vocab)))
            )
        )


def interactive(vocab, embeddings, args):
    while True:
        print("\n1 查看 token 向量 | 2 比较两个 token | 3 查找最相近 token")
        print("4 多个 token 的比较矩阵 | 5 查看词表 | 0 退出")
        choice = input("请选择：").strip()
        try:
            if choice in {"0", "q", "/q"}:
                return
            if choice == "1":
                show_vector(vocab, embeddings, input("输入 token 或 ID："))
            elif choice == "2":
                pair = input("输入两个 token 或 ID，用空格隔开，例如 cat dog：").split()
                if len(pair) != 2:
                    raise ValueError("需要输入两个 token 或 ID")
                show_comparison(vocab, embeddings, *pair)
            elif choice == "3":
                token = input("输入 token 或 ID：")
                metric = (
                    input(f"排序指标 cosine/l2/l1/dot [{args.metric}]：")
                    .strip()
                    .lower()
                    or args.metric
                )
                count = input(f"返回数量 [{args.top_k}]：").strip()
                show_neighbors(
                    vocab,
                    embeddings,
                    token,
                    int(count) if count else args.top_k,
                    metric,
                    args.include_special,
                )
            elif choice == "4":
                selected = input(
                    "输入多个 token 或 ID，用空格隔开，例如 cat dog i you："
                ).split()
                show_matrices(vocab, embeddings, selected)
            elif choice == "5":
                show_vocab(vocab)
            else:
                print("请输入菜单中的数字。")
        except ValueError as error:
            print("输入有误：", error)


def main():
    parser = argparse.ArgumentParser(description="查看 MiniGPT 词向量、距离和相似度")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "best.pt",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--token", help="查看一个 token 的完整向量，也可输入 ID")
    action.add_argument(
        "--compare", nargs=2, metavar=("TOKEN_A", "TOKEN_B"), help="比较两个 token"
    )
    action.add_argument("--neighbors", help="查找与指定 token 最相近的 token")
    action.add_argument("--matrix", nargs="+", help="显示指定 token 之间的余弦/L2 矩阵")
    action.add_argument("--vocab", action="store_true", help="显示完整词表")
    parser.add_argument("--top-k", type=int, default=10, help="近邻数量，默认 10")
    parser.add_argument(
        "--metric", choices=METRICS, default="cosine", help="近邻排序指标"
    )
    parser.add_argument(
        "--include-special", action="store_true", help="近邻候选包含特殊 token"
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k 必须至少为 1")

    try:
        vocab, embeddings = load_embeddings(args.checkpoint)
        if args.token is not None:
            show_vector(vocab, embeddings, args.token)
        elif args.compare is not None:
            show_comparison(vocab, embeddings, *args.compare)
        elif args.neighbors is not None:
            show_neighbors(
                vocab,
                embeddings,
                args.neighbors,
                args.top_k,
                args.metric,
                args.include_special,
            )
        elif args.matrix is not None:
            show_matrices(vocab, embeddings, args.matrix)
        elif args.vocab:
            show_vocab(vocab)
        else:
            interactive(vocab, embeddings, args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n已结束。")
