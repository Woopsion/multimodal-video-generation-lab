"""Day 12：最小真实 VLM 图像问答脚本。

本文件是学习用脚本，不是生产推理服务。它的目标是把 Day12 的 VLM 主链路跑通，
并在代码里明确标出哪些部分当前必须深入掌握，哪些部分先理解即可。

学习标注说明：

【需要深入掌握】
    Day12 当前最重要的代码路径。你需要能说清楚它在 VLM 中承担什么角色，
    输入输出是什么形状，为什么它处在这一步。

【理解即可】
    工程辅助代码。你不需要现在背实现细节，只要知道它在帮助脚本更可控、
    更容易复现或更方便观察结果。

本脚本的核心链路：

image path + question text
-> AutoProcessor / Idefics3Processor
-> pixel_values / input_ids / attention_mask
-> AutoModelForImageTextToText.generate()
-> processor.batch_decode()
-> natural-language answer

换成 Day12 的概念语言：

原始图片 + 用户问题
-> processor 把图片和文本整理成模型输入
-> vision encoder 读取 pixel_values
-> projector 把视觉特征对齐到 LLM hidden size
-> LLM decoder 自回归生成 answer tokens
-> decode 得到最终自然语言回答

默认策略：离线优先、本地优先。只有显式传入 --allow-download 时，才允许
Transformers 从 Hugging Face 下载权重。这样能避免学习脚本静默联网下载大文件。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


# 【需要深入掌握】这里选择的是 Day12 的轻量 VLM 权重。
# 这个模型足够小，可以在本机 CUDA 上快速跑通完整 VLM 闭环；它不是为了追求最好效果，
# 而是为了让你清楚观察 processor、张量形状、generate、decode 和失败边界。
DEFAULT_MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

# 【理解即可】如果你手动把模型权重保存到项目目录下的 pretrained/，脚本会优先用本地路径。
# 当前真实运行主要依赖 Hugging Face 的本机缓存，不建议把大模型权重提交到 Git 仓库。
DEFAULT_LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parent
    / "pretrained"
    / "HuggingFaceTB-SmolVLM-256M-Instruct"
)

# 【理解即可】这两个路径用于教学样例：自动生成三张测试图和对应 JSONL 任务清单。
DEFAULT_ASSET_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_TASKS_PATH = Path(__file__).resolve().parent / "sample_tasks.jsonl"


def default_model_reference() -> str:
    """【理解即可】选择模型来源：优先项目本地目录，否则使用 Hugging Face 模型名。"""
    if DEFAULT_LOCAL_MODEL_DIR.exists():
        return str(DEFAULT_LOCAL_MODEL_DIR)
    return DEFAULT_MODEL_NAME


def section(title: str) -> None:
    """【理解即可】打印阶段标题，让命令行输出更适合学习观察。"""
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def line() -> None:
    """【理解即可】打印分隔线，只影响输出可读性。"""
    print("-" * 96)


def choose_device(requested: str) -> Any:
    """【需要深入掌握】选择推理设备。

    你需要理解：真实 VLM 推理成本明显高于 CLIP 检索 toy demo，所以脚本必须知道
    当前是在 CPU、CUDA 还是 MPS 上运行。设备会直接影响速度、显存和 dtype 选择。
    """
    import torch

    # 【理解即可】如果用户明确传 --device cpu/cuda/mps，就尊重用户选择。
    if requested != "auto":
        return torch.device(requested)

    # 【需要深入掌握】优先使用 CUDA，因为本次 SmolVLM 实验就是在 CUDA 上观察显存和耗时。
    if torch.cuda.is_available():
        return torch.device("cuda")

    # 【理解即可】macOS 上可能有 MPS；Windows 本机通常不会走这里。
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # 【理解即可】兜底到 CPU。CPU 能跑但会慢很多。
    return torch.device("cpu")


def tensor_shape(name: str, value: Any, note: str | None = None) -> None:
    """【需要深入掌握】打印关键张量的 shape / dtype / device。

    Day12 不是只看答案文本，而是要观察 processor 输出了什么：
    - input_ids：文本 token 和图像占位相关 token
    - attention_mask：哪些 token 是有效上下文
    - pixel_values：送进 vision encoder 的图片张量
    - generated_ids：LLM 生成后的 token id 序列
    """
    if not hasattr(value, "shape"):
        return
    line()
    print(f"{name}.shape = {tuple(value.shape)}")
    print(f"{name}.dtype = {value.dtype}")
    print(f"{name}.device = {value.device}")
    if note:
        print(note)


def load_transformers_classes() -> tuple[Any, Any]:
    """【需要深入掌握】加载 Transformers 的自动类。

    AutoProcessor：负责把图片和文本整理成模型输入。
    AutoModelForImageTextToText：负责加载真正的 VLM，并执行 generate。

    这里用 Any 是为了让脚本保持通用教学风格：不同 Transformers 版本、不同 VLM
    可能返回不同具体类名。本次实测会得到 Idefics3Processor 和 Idefics3ForConditionalGeneration。
    """
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ModuleNotFoundError:
        print(
            "The Python package 'transformers' is not installed.\n"
            "Install it with: uv add transformers safetensors pillow"
        )
        raise SystemExit(1)
    return AutoProcessor, AutoModelForImageTextToText


def load_image(path: str | Path) -> Image.Image:
    """【需要深入掌握】把图片路径变成 RGB 图片对象。

    对 VLM 来说，用户传入的是文件路径，但模型不能直接吃路径字符串。
    路径必须先读成图片，再交给 processor 转成 pixel_values。
    """
    image_path = Path(path)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        raise SystemExit(1)

    # 【需要深入掌握】统一转 RGB，避免 PNG 透明通道、灰度图等格式差异影响 processor。
    return Image.open(image_path).convert("RGB")


def load_tasks(path: str | Path) -> list[dict[str, str]]:
    """【理解即可】读取 JSONL 测试任务。

    JSONL 的每一行是一条测试样例，包含：
    id / image / question / expected。

    这部分不是 VLM 模型核心，但它让我们能系统地跑场景描述、OCR、图表理解三类任务，
    并把模型回答和人工期望答案对比。
    """
    task_path = Path(path)
    if not task_path.exists():
        print(f"Task file not found: {task_path}")
        raise SystemExit(1)

    tasks: list[dict[str, str]] = []
    for line_no, raw_line in enumerate(task_path.read_text(encoding="utf-8").splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            task = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON on line {line_no}: {exc}")
            raise SystemExit(1) from exc

        # 【理解即可】做最小字段校验，避免任务文件少字段时后面报错不清楚。
        for field in ("id", "image", "question", "expected"):
            if field not in task:
                print(f"Task line {line_no} misses field: {field}")
                raise SystemExit(1)
        tasks.append(task)
    return tasks


def move_inputs_to_device(inputs: Any, device: Any) -> Any:
    """【需要深入掌握】把 processor 产生的张量移动到模型所在设备。

    如果模型在 cuda，但 input_ids / pixel_values 还在 cpu，PyTorch 会报 device mismatch。
    所以真实推理脚本必须保证：模型参数和输入张量在同一个 device 上。
    """
    # Transformers 的 BatchFeature 通常自带 .to(device)，可以一次性移动全部张量。
    if hasattr(inputs, "to"):
        return inputs.to(device)

    # 【理解即可】兜底路径：如果 inputs 是普通 dict，就逐个移动里面的 tensor。
    return {name: value.to(device) if hasattr(value, "to") else value for name, value in inputs.items()}


def build_prompt(processor: Any, question: str) -> str:
    """【需要深入掌握】构造 VLM 对话 prompt。

    关键点：<image> 不是图片内容本身，而是提示词模板中的图片插槽。
    真正的图片内容来自 processor 生成的 pixel_values，再由模型内部 vision encoder 读取。
    """
    messages = [
        {
            "role": "user",
            "content": [
                # 【需要深入掌握】这里声明“这一轮用户消息里有一张图片”。
                # apply_chat_template 会把它转成该模型认识的图像占位格式。
                {"type": "image"},
                # 【需要深入掌握】用户的自然语言问题。它会被 tokenizer 转成 input_ids。
                {"type": "text", "text": question},
            ],
        }
    ]

    # 【需要深入掌握】优先使用模型自己的 chat template。
    # 不同 VLM 的特殊 token、角色名、结束符都可能不同，硬写字符串容易格式不匹配。
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, add_generation_prompt=True)

    # 【理解即可】兜底格式。真实项目中应尽量使用模型官方模板。
    return f"User: <image>\n{question}\nAssistant:"


def encode_inputs(processor: Any, image: Image.Image, prompt: str) -> Any:
    """【需要深入掌握】processor 把图片和文本变成模型可用张量。

    这是 VLM 推理脚本里最重要的分界线之一：
    - 进入这个函数之前：图片还是 PIL.Image，问题还是字符串。
    - 走出这个函数之后：图片和文本已经变成 PyTorch 张量。

    本次 SmolVLM 会产生类似：
    - input_ids / attention_mask
    - pixel_values / pixel_attention_mask
    """
    try:
        # 【需要深入掌握】许多 processor 支持批量图片，所以这里用 images=[image]。
        return processor(images=[image], text=prompt, return_tensors="pt")
    except Exception:
        # 【理解即可】兼容少数 processor 只接受单张图片对象的接口差异。
        return processor(images=image, text=prompt, return_tensors="pt")


def decode_answer(processor: Any, generated_ids: Any, input_token_count: int | None) -> str:
    """【需要深入掌握】把生成的 token ids 解码成答案文本。

    generate() 通常返回“输入 token + 新生成 token”。如果直接 decode 全部，可能会把 prompt
    也打印出来。学习阶段我们更关心模型新生成的回答，所以要把输入长度切掉。
    """
    if input_token_count is not None and generated_ids.ndim == 2:
        # 【需要深入掌握】只保留新生成 token，也就是 answer tokens。
        new_ids = generated_ids[:, input_token_count:]
    else:
        # 【理解即可】兜底：如果无法确定输入长度，就 decode 全部。
        new_ids = generated_ids

    # 【需要深入掌握】batch_decode 是 tokenizer decode 的批量版本；skip_special_tokens 会去掉特殊控制符。
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def maybe_cuda_memory_mb(device: Any) -> float | None:
    """【理解即可】读取 CUDA 峰值显存，帮助记录轻量模型的运行边界。"""
    import torch

    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def reset_cuda_memory(device: Any) -> None:
    """【理解即可】每次生成前重置 CUDA 峰值显存统计，便于观察单次任务成本。"""
    import torch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def load_processor_and_model(args: argparse.Namespace) -> tuple[Any, Any, Any, bool]:
    """【需要深入掌握】加载 processor、模型和设备。

    这是完整 VLM 推理的准备阶段：
    1. 加载 AutoProcessor：获得图片/文本预处理器和 chat template。
    2. 加载 AutoModelForImageTextToText：获得真实 VLM 权重。
    3. 选择 device / dtype：决定推理在哪里运行、用什么精度。
    4. model.eval()：切换到推理模式。
    """
    import torch

    AutoProcessor, AutoModelForImageTextToText = load_transformers_classes()
    device = choose_device(args.device)

    # 【需要深入掌握】local_files_only 是学习脚本的安全边界。
    # 默认不下载；只有用户明确传 --allow-download，才允许访问 Hugging Face Hub。
    local_files_only = not args.allow_download

    section("Step 01: load VLM processor and model")
    print(f"model name/path = {args.model}")
    print(f"device = {device}")
    print(f"local_files_only = {local_files_only}")
    print(
        "\nThis script is offline/local-only by default. If the model is not already\n"
        "available locally or in the Hugging Face cache, re-run with --allow-download\n"
        "only when you are ready to download remote files."
    )

    try:
        # 【需要深入掌握】processor 不等于模型。
        # processor 负责输入整理：图片 resize/normalize、文本 tokenize、chat template。
        processor = AutoProcessor.from_pretrained(
            args.model,
            local_files_only=local_files_only,
            trust_remote_code=args.trust_remote_code,
        )

        # 【需要深入掌握】dtype 决定模型参数精度。
        # CUDA 上默认 bfloat16，是为了降低显存，同时保持比 float16 更稳的数值范围。
        dtype = torch.bfloat16 if device.type == "cuda" and args.dtype == "auto" else None
        if args.dtype == "float16":
            dtype = torch.float16
        elif args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float32":
            dtype = torch.float32

        model_kwargs: dict[str, Any] = {
            # 【需要深入掌握】这两个参数控制“是否只用本地缓存”和“是否信任远程自定义代码”。
            "local_files_only": local_files_only,
            "trust_remote_code": args.trust_remote_code,
        }
        if dtype is not None:
            model_kwargs["dtype"] = dtype

        # 【需要深入掌握】这里真正加载 VLM 权重。
        # 本次 SmolVLM 会实例化为 Idefics3ForConditionalGeneration。
        model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    except OSError as exc:
        # 【理解即可】常见失败：本地没缓存模型，且没有传 --allow-download。
        print("\nModel/processor loading failed.")
        print(f"Reason: {exc}")
        print(
            "\nCommon fix when you allow a first download:\n"
            f"  uv run python {sys.argv[0]} --model {DEFAULT_MODEL_NAME} --allow-download\n"
            "\nThe default is intentionally offline/local-only so this learning script does\n"
            "not download model files without your explicit choice."
        )
        raise SystemExit(1) from exc

    # 【需要深入掌握】模型参数移动到目标设备；输入张量后面也必须移动到同一设备。
    model.to(device)

    # 【需要深入掌握】eval() 关闭训练相关行为，例如 dropout。推理脚本必须调用。
    model.eval()

    # 【理解即可】打印具体类名，帮助把 Auto 类映射到真实实现。
    print(f"processor class = {processor.__class__.__name__}")
    print(f"model class = {model.__class__.__name__}")
    print(f"model dtype parameter sample = {next(model.parameters()).dtype}")
    print(f"model config type = {model.config.__class__.__name__}")
    return processor, model, device, local_files_only


def run_one_question(
    *,
    processor: Any,
    model: Any,
    device: Any,
    image_path: str | Path,
    question: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """【需要深入掌握】执行一次“单图 + 单问题 -> 文本答案”的 VLM 推理。

    这是整份脚本最核心的函数。你需要能复述它的顺序：

    1. load_image：路径 -> PIL.Image
    2. build_prompt：问题 -> 模型对话模板
    3. encode_inputs：图片 + prompt -> 张量
    4. move_inputs_to_device：张量 -> CUDA/CPU
    5. model.generate：自回归生成 token ids
    6. decode_answer：token ids -> 自然语言答案
    7. 记录耗时、显存和回答边界
    """
    import torch

    # 【需要深入掌握】图片从路径变成 RGB 图像对象。
    image = load_image(image_path)

    # 【需要深入掌握】问题文本被包装成模型认识的对话格式，里面包含图片占位。
    prompt = build_prompt(processor, question)

    section("Step 02: processor(image, question) -> model tensors")
    print(f"image path = {image_path}")
    print(f"image.size = {image.size}")
    print(f"question = {question!r}")
    print("prompt from chat template:")
    print(prompt)

    # 【需要深入掌握】这是 VLM 输入链路的关键一步。
    # processor 会同时处理图片和文本，输出模型需要的张量。
    encoded = encode_inputs(processor, image, prompt)

    # 【需要深入掌握】确保 input_ids、pixel_values 等张量和 model 在同一个 device。
    encoded = move_inputs_to_device(encoded, device)

    input_token_count = None
    if "input_ids" in encoded:
        # 【需要深入掌握】记录输入 token 长度，后面 decode 时用来切掉 prompt，只保留 answer tokens。
        input_token_count = int(encoded["input_ids"].shape[-1])
        tensor_shape("input_ids", encoded["input_ids"], "[B, T]: text token IDs plus image placeholders.")
    if "attention_mask" in encoded:
        tensor_shape("attention_mask", encoded["attention_mask"], "[B, T]: 1 means visible token, 0 means padding.")
    if "pixel_values" in encoded:
        # 【需要深入掌握】pixel_values 是送给 vision encoder 的图片张量，不是原始 PNG/JPG 文件。
        tensor_shape("pixel_values", encoded["pixel_values"], "Image tensor prepared for the vision encoder.")
    if "pixel_attention_mask" in encoded:
        # 【理解即可】某些 VLM 会为图片 patch 也准备 mask，用于处理图片切块后的有效区域。
        tensor_shape("pixel_attention_mask", encoded["pixel_attention_mask"], "Image patch mask prepared by the processor.")

    section("Step 03: model.generate -> output token ids")
    reset_cuda_memory(device)
    start = time.perf_counter()

    # 【需要深入掌握】inference_mode 表示纯推理，不构建反向传播图，速度更快、显存更省。
    with torch.inference_mode():
        # 【需要深入掌握】generate 是自回归生成入口。
        # 模型基于 image tokens + text tokens 的上下文，一步一步预测 answer tokens。
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            # 【理解即可】do_sample=False 表示贪心/确定性生成，便于学习阶段复现结果。
            do_sample=False,
        )
    elapsed = time.perf_counter() - start

    tensor_shape("generated_ids", generated_ids, "[B, input_tokens + answer_tokens].")

    # 【需要深入掌握】把 generated_ids 中的新 token 部分 decode 成答案。
    answer = decode_answer(processor, generated_ids, input_token_count)
    peak_memory_mb = maybe_cuda_memory_mb(device)

    section("Step 04: decode answer and record boundary")
    print(f"answer = {answer!r}")
    print(f"elapsed_seconds = {elapsed:.3f}")
    print(f"peak_cuda_memory_mb = {peak_memory_mb if peak_memory_mb is not None else 'n/a'}")

    # 【理解即可】返回结构化结果，方便 JSONL 保存和后续复盘。
    return {
        "image": str(image_path),
        "question": question,
        "answer": answer,
        "elapsed_seconds": round(elapsed, 3),
        "peak_cuda_memory_mb": None if peak_memory_mb is None else round(peak_memory_mb, 1),
    }


def load_font(size: int) -> ImageFont.ImageFont:
    """【理解即可】加载系统字体，用于生成教学样例图。"""
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def create_sample_images(asset_dir: Path = DEFAULT_ASSET_DIR, tasks_path: Path = DEFAULT_TASKS_PATH) -> None:
    """【理解即可】生成三类本地测试图片和 JSONL 任务清单。

    这部分代码不是 VLM 推理核心，但它服务于 Day12 的评测思路：
    - 场景描述：测试物体、位置和关系
    - OCR / 截图：测试文字识别和界面布局
    - 图表理解：测试数值比较和趋势理解

    你当前不需要深入掌握 PIL 绘图 API，只要理解为什么要准备这三类样例。
    """
    asset_dir.mkdir(parents=True, exist_ok=True)
    font_large = load_font(34)
    font_mid = load_font(24)
    font_small = load_font(18)

    # 【理解即可】样例 1：场景描述图。用于观察模型能否识别物体和相对位置。
    scene_path = asset_dir / "scene_desk.png"
    image = Image.new("RGB", (720, 480), "#f8f4ea")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 300, 660, 410), fill="#c9a26a", outline="#6b4a24", width=4)
    draw.rectangle((250, 150, 470, 295), fill="#2f3a3a", outline="#111", width=4)
    draw.rectangle((270, 170, 450, 270), fill="#bfe7ff")
    draw.rectangle((220, 295, 500, 320), fill="#242424")
    draw.ellipse((520, 235, 610, 325), fill="#ffffff", outline="#6b4a24", width=4)
    draw.ellipse((544, 258, 586, 294), fill="#8b4a2f")
    draw.arc((598, 258, 650, 306), start=270, end=90, fill="#6b4a24", width=5)
    draw.rectangle((105, 220, 220, 330), fill="#f6f0c4", outline="#6b6b35", width=3)
    draw.line((120, 248, 205, 248), fill="#9d8b45", width=3)
    draw.line((120, 276, 205, 276), fill="#9d8b45", width=3)
    draw.line((135, 325, 230, 255), fill="#c62828", width=8)
    draw.text((70, 38), "Desk scene", fill="#202427", font=font_large)
    draw.text((70, 82), "Laptop, notebook, pen, and coffee cup", fill="#555", font=font_mid)
    image.save(scene_path)

    # 【理解即可】样例 2：OCR / UI 截图图。用于观察模型能否读对指定文字。
    ocr_path = asset_dir / "ocr_ui.png"
    image = Image.new("RGB", (720, 480), "#edf4ef")
    draw = ImageDraw.Draw(image)
    draw.rectangle((95, 65, 625, 415), fill="#fffaf0", outline="#2f3836", width=5)
    draw.text((150, 100), "MODEL LAB", fill="#0f766e", font=font_large)
    draw.text((150, 160), "Status: READY", fill="#202427", font=font_mid)
    draw.text((150, 210), "Input image loaded", fill="#65706c", font=font_mid)
    draw.rectangle((150, 285, 390, 350), fill="#b4532a", outline="#7a2f12", width=3)
    draw.text((184, 302), "RUN DEMO", fill="#ffffff", font=font_mid)
    draw.text((150, 378), "Footer: Day12 VLM QA", fill="#65706c", font=font_small)
    image.save(ocr_path)

    # 【理解即可】样例 3：图表理解图。用于观察模型能否同时回答最高值和最低值。
    chart_path = asset_dir / "chart_sales.png"
    image = Image.new("RGB", (720, 480), "#fffaf0")
    draw = ImageDraw.Draw(image)
    draw.text((70, 40), "Quarterly Demo Scores", fill="#202427", font=font_large)
    origin = (100, 390)
    draw.line((origin[0], 90, origin[0], origin[1]), fill="#202427", width=4)
    draw.line((origin[0], origin[1], 650, origin[1]), fill="#202427", width=4)
    values = [("Q1", 3), ("Q2", 5), ("Q3", 2), ("Q4", 7)]
    colors = ["#0f766e", "#5b4b8a", "#b4532a", "#2f3836"]
    max_value = 8
    bar_width = 70
    gap = 62
    for index, ((label, value), color) in enumerate(zip(values, colors, strict=True)):
        x0 = 145 + index * (bar_width + gap)
        bar_height = int((value / max_value) * 260)
        y0 = origin[1] - bar_height
        draw.rectangle((x0, y0, x0 + bar_width, origin[1]), fill=color)
        draw.text((x0 + 18, origin[1] + 16), label, fill="#202427", font=font_mid)
        draw.text((x0 + 22, y0 - 32), str(value), fill="#202427", font=font_mid)
    draw.text((70, 424), "Q4 is tallest; Q3 is lowest.", fill="#65706c", font=font_mid)
    image.save(chart_path)

    # 【需要深入掌握】expected 是人工期望答案。
    # 它不是给模型看的，而是给我们评估模型回答边界用的。
    tasks = [
        {
            "id": "scene_desk",
            "type": "scene",
            "image": str(scene_path),
            "question": "What objects are on the desk, and where is the coffee cup relative to the laptop?",
            "expected": "Laptop in the center, notebook and pen on the left, coffee cup to the right of the laptop.",
        },
        {
            "id": "ocr_ui",
            "type": "ocr",
            "image": str(ocr_path),
            "question": "What is the button label and what is the status text?",
            "expected": "Button label: RUN DEMO. Status text: READY.",
        },
        {
            "id": "chart_sales",
            "type": "chart",
            "image": str(chart_path),
            "question": "Which quarter has the highest score, and which quarter has the lowest score?",
            "expected": "Q4 has the highest score. Q3 has the lowest score.",
        },
    ]

    # 【理解即可】写成 JSONL，便于逐行读取、追加和保存实验结果。
    tasks_path.write_text("\n".join(json.dumps(task, ensure_ascii=False) for task in tasks) + "\n", encoding="utf-8")

    section("Created Day12 sample images and task file")
    print(f"asset_dir = {asset_dir}")
    print(f"tasks_path = {tasks_path}")
    for task in tasks:
        print(f"- {task['id']}: {task['image']}")


def parse_args() -> argparse.Namespace:
    """【需要深入掌握】定义命令行输入。

    最小 image QA 脚本的核心输入是：
    - --image：图片路径
    - --question：问题文本

    批量测试时使用：
    - --tasks：JSONL 任务文件
    - --output：保存结果 JSONL
    """
    parser = argparse.ArgumentParser(description="Day 12: run minimal real VLM image QA.")
    parser.add_argument("--model", default=default_model_reference(), help="Model name or local model path.")
    parser.add_argument("--image", default=None, help="Image path for a single question.")
    parser.add_argument("--question", default=None, help="Question text for a single image.")
    parser.add_argument("--tasks", default=None, help="JSONL task file with id, image, question, expected.")
    parser.add_argument("--output", default=None, help="Optional JSONL path for saved results.")
    parser.add_argument("--create-samples", action="store_true", help="Create three local sample images and sample_tasks.jsonl.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, or mps.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="Model dtype.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Maximum new answer tokens.")
    parser.add_argument("--allow-download", action="store_true", help="Allow downloading model files if not cached locally.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to Transformers.")
    return parser.parse_args()


def main() -> None:
    """【需要深入掌握】脚本入口，负责把所有函数串成完整流程。"""
    args = parse_args()

    # 【理解即可】只生成样例图和任务文件，不跑模型。
    if args.create_samples:
        create_sample_images()
        if args.image is None and args.tasks is None:
            return

    # 【需要深入掌握】脚本必须至少知道“要问哪张图、问什么问题”。
    # 单图模式：--image + --question
    # 批量模式：--tasks
    if args.tasks is None and (args.image is None or args.question is None):
        print(
            "Provide either:\n"
            "  --image PATH --question TEXT\n"
            "or:\n"
            "  --tasks experiments/vlm-qa/sample_tasks.jsonl\n"
            "\nTo create sample images first:\n"
            "  uv run python experiments/vlm-qa/image_qa.py --create-samples"
        )
        raise SystemExit(1)

    # 【需要深入掌握】先加载 processor 和 model，再开始问答。
    processor, model, device, _ = load_processor_and_model(args)

    results: list[dict[str, Any]] = []
    if args.tasks:
        # 【需要深入掌握】批量任务并不是模型特殊能力，只是循环调用单图单问答。
        tasks = load_tasks(args.tasks)
        for index, task in enumerate(tasks, start=1):
            section(f"Task {index}/{len(tasks)}: {task['id']} ({task.get('type', 'unknown')})")
            print(f"expected = {task['expected']}")
            result = run_one_question(
                processor=processor,
                model=model,
                device=device,
                image_path=task["image"],
                question=task["question"],
                max_new_tokens=args.max_new_tokens,
            )

            # 【需要深入掌握】把人工期望答案和模型答案放在同一条记录里，方便分析失败边界。
            result.update({"id": task["id"], "type": task.get("type"), "expected": task["expected"]})
            results.append(result)
    else:
        # 【需要深入掌握】单图单问答是整个脚本的最小闭环。
        results.append(
            run_one_question(
                processor=processor,
                model=model,
                device=device,
                image_path=args.image,
                question=args.question,
                max_new_tokens=args.max_new_tokens,
            )
        )

    section("Summary")
    for result in results:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    # 【理解即可】保存 JSONL，便于后续把多次实验结果进行对比。
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(json.dumps(result, ensure_ascii=False) for result in results) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
