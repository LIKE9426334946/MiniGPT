"""用完全未训练的 MiniGPT，逐个观察 token 的生成过程。

放在项目根目录，与 model.py、generate.py、vocab.py、config.yaml 同级。
运行：python generate_untrained.py
自动运行：python generate_untrained.py --prompt "i love" --auto
换一组初始参数：python generate_untrained.py --prompt "i love" --seed 42

本脚本只新建模型，不读取 checkpoint，也不执行训练。
复用 generate.py 的 generate_steps，因此逐步输出和停止规则保持一致。
"""

import argparse
from pathlib import Path

import torch
import yaml

from generate import generate_steps
from model import MiniGPT


def create_untrained_model(config_path, device_name="auto", seed=None):
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError("配置文件需要包含 model 配置")

    # 默认使用训练配置中的种子，方便对照训练前后的结果。
    if seed is None:
        seed = config.get("train", {}).get("seed", 42)

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA，请使用 --device cpu")

    # 1. 先固定随机种子，再创建模型。
    torch.manual_seed(seed)

    # 2. __init__ 会创建 embedding、Q/K/V、前馈网络和输出层等。
    #    各层全部保留 PyTorch 的默认初始化，没有加载任何训练后的权重。
    #    注意：<pad> embedding 默认全零，LayerNorm 的缩放/偏置默认是 1/0。
    model = MiniGPT(config)

    # 3. 仅移动设备并设为评估模式；eval() 不会训练模型。
    model.to(device)
    model.eval()

    print("模型状态：使用新建模型的初始参数，尚未训练")
    print(f"配置文件：{config_path}")
    print(f"随机种子：{seed}，设备：{device}")
    print(f"token embedding 形状：{tuple(model.token_embedding.weight.shape)}")
    print(f"最大输入长度：{model.max_seq_len}（包含 <bos>）")
    print("模型只在启动时初始化一次；所有后续 token 和前缀共用这组参数。")
    return model, device


def main():
    parser = argparse.ArgumentParser(description="逐步观察未训练 MiniGPT 的 token 预测")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="模型配置，默认读取本脚本旁边的 config.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="初始化种子，默认使用 config.yaml 中的 train.seed",
    )
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda")
    parser.add_argument("--prompt", default=None, help="直接指定输入前缀，例如 i love")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--auto", action="store_true", help="自动逐步输出，不等待回车")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens 必须至少为 1")

    try:
        model, device = create_untrained_model(args.config, args.device, args.seed)

        # generate_steps 内部使用 torch.inference_mode()，只做前向传播。
        # 每一步：完整前缀 -> 最后一个位置的 logits -> argmax -> 追加新 token。
        if args.prompt is not None:
            generate_steps(
                model, device, args.prompt, args.max_new_tokens, not args.auto
            )
            return

        print("\n输入英文前缀，例如 i love；直接回车从 <bos> 开始，输入 /q 退出。")
        while True:
            prefix = input("\n请输入前几个 token：")
            if prefix.strip().lower() == "/q":
                break
            try:
                generate_steps(
                    model, device, prefix, args.max_new_tokens, not args.auto
                )
            except ValueError as error:
                print("输入有误：", error)
    except (OSError, KeyError, ValueError, RuntimeError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n已结束。")
