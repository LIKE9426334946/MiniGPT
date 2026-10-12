import argparse
import re
from pathlib import Path

import torch

from model import MiniGPT
from vocab import id_to_token, token_to_id, tokens


def load_model(checkpoint_path, device_name):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"找不到模型文件：{checkpoint_path}\n"
            "请先运行 train.py，或通过 --checkpoint 指定 best.pt 的路径。"
        )

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA，请使用 --device cpu")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"config", "model_state_dict", "vocab"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("模型文件格式不匹配，请使用上一版 train.py 保存的 best.pt")
    if checkpoint["vocab"] != list(tokens):
        raise ValueError("当前 vocab.py 的 token 内容或顺序与训练时不同")

    # 使用训练时保存的配置创建模型，确保网络结构与权重一致。
    model = MiniGPT(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"已加载：{checkpoint_path}")
    print(f"设备：{device}，最大输入长度：{model.max_seq_len}（包含 <bos>）")
    return model, device


def encode_prefix(text):
    # 转为小写，并把标点分开，例如 "cat." -> "cat ."。
    text = re.sub(r"([.,?!])", r" \1 ", text.strip().lower())
    words = text.split()
    # 允许手动写一个开头的 <bos>，不会重复添加。
    if words and words[0] == "<bos>":
        words = words[1:]

    special_tokens = {"<pad>", "<unk>", "<bos>", "<eos>"}
    if any(word in special_tokens for word in words):
        raise ValueError("前缀中请填写普通单词和标点；程序会自动处理边界 token")
    unknown = list(dict.fromkeys(w for w in words if w not in token_to_id))
    if unknown:
        raise ValueError(
            "这些词不在当前 100 个 token 的词表中："
            + ", ".join(unknown)
            + "。可以先试试 i love、the cat、where is。"
        )

    # 前缀只添加 <bos>；<eos> 由模型自己预测。
    return [token_to_id["<bos>"]] + [token_to_id[word] for word in words]


def decode_ids(ids):
    return " ".join(id_to_token[i] for i in ids)


@torch.inference_mode()
def generate_steps(model, device, prefix, max_new_tokens=20, step_by_step=True):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须至少为 1")
    ids = encode_prefix(prefix)
    if len(ids) > model.max_seq_len:
        raise ValueError(
            f"前缀加上 <bos> 后有 {len(ids)} 个 token，"
            f"超过最大输入长度 {model.max_seq_len}"
        )

    model.eval()
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)  # [1, T]
    prefix_length = input_ids.size(1)
    stop_reason = f"达到最大新增 token 数 {max_new_tokens}"

    for step in range(1, max_new_tokens + 1):
        print(f"\n第 {step} 步")
        print("本轮输入 token：", decode_ids(input_ids[0].tolist()))
        print("本轮输入 ID：", input_ids[0].tolist())
        print("本轮输入形状：", tuple(input_ids.shape))

        if step_by_step:
            command = input("按回车生成一个 token，输入 /q 停止本次续写：").strip()
            if command.lower() == "/q":
                stop_reason = "你停止了本次续写"
                break

        # 1. 每一轮都把完整的当前序列送入模型。
        #    这里不使用 KV cache，便于观察完整的前向传播。
        logits = model(input_ids)  # [1, T, 100]

        # 2. 只取最后一个位置的输出，用它预测紧接着的下一个 token。
        next_logits = logits[:, -1, :]  # [1, 100]
        if not torch.isfinite(next_logits).all():
            raise RuntimeError("模型输出包含 NaN 或 Inf，请检查保存的模型")
        probabilities = torch.softmax(next_logits, dim=-1)

        # 3. 贪心选择：取分数最高的 token ID。
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # [1, 1]
        next_id = next_token.item()

        print("模型输出 logits 形状：", tuple(logits.shape))
        print("取最后一个位置后：", tuple(next_logits.shape))

        # Top 5 仅用于展示候选；实际选择由上面的 argmax 完成。
        top_probs, top_ids = torch.topk(
            probabilities[0], k=min(5, probabilities.size(-1))
        )
        candidates = [
            f"{id_to_token[i]}(ID={i}, 概率={p:.2%})"
            for p, i in zip(top_probs.tolist(), top_ids.tolist())
        ]
        print("概率最高的候选：", "；".join(candidates))
        print(
            f"本轮输出 token：{id_to_token[next_id]} | ID={next_id} | "
            f"概率={probabilities[0, next_id].item():.2%}"
        )

        # 4. 把新 token 拼到原输入后面，下一轮用更新后的整个序列。
        input_ids = torch.cat([input_ids, next_token], dim=1)  # [1, T+1]
        print("追加后的序列：", decode_ids(input_ids[0].tolist()))
        print("追加后的 ID：", input_ids[0].tolist())

        if next_id == token_to_id["<eos>"]:
            stop_reason = "模型输出了 <eos>，本次续写结束"
            break
        if input_ids.size(1) > model.max_seq_len:
            # 最后一步可以预测一个新 token，但更长的序列不能再次输入。
            # 保留完整上下文，达到上限后停止，不对前缀进行截断。
            stop_reason = (
                f"下一轮需要输入 {input_ids.size(1)} 个 token，"
                f"超过模型上限 {model.max_seq_len}，停止继续生成"
            )
            break

    result_ids = input_ids[0].tolist()
    # 最终文本去掉开头 <bos> 和结尾 <eos>，补回标点格式。
    text_ids = result_ids[1:]
    if text_ids and text_ids[-1] == token_to_id["<eos>"]:
        text_ids = text_ids[:-1]
    result_text = re.sub(r"\s+([.,?!])", r"\1", decode_ids(text_ids))
    print("\n停止原因：", stop_reason)
    print("新增 token 数（包含生成的 <eos>）：", len(result_ids) - prefix_length)
    print("最终文本：", result_text)
    return result_ids


def main():
    parser = argparse.ArgumentParser(description="逐步观察 MiniGPT 的 token 生成过程")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "best.pt",
        help="训练保存的模型文件路径",
    )
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda")
    parser.add_argument("--prompt", default=None, help="直接指定输入前缀")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--auto", action="store_true", help="自动逐步输出，不等待回车")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens 必须至少为 1")

    model, device = load_model(args.checkpoint, args.device)
    if args.prompt is not None:
        generate_steps(model, device, args.prompt, args.max_new_tokens, not args.auto)
        return

    print("\n输入英文前缀，例如 i love。直接回车可从 <bos> 开始，输入 /q 退出。")
    while True:
        prefix = input("\n请输入前几个 token：")
        if prefix.strip().lower() == "/q":
            break
        try:
            generate_steps(model, device, prefix, args.max_new_tokens, not args.auto)
        except ValueError as error:
            print("输入有误：", error)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n已结束。")
