"""Minimal CLIP image-search demo for Day 10.

Run:
  .venv/bin/python demos/clip-image-search/app.py

The server is offline/local-only by default. Pass --allow-download explicitly if
the CLIP model is not already cached and you are ready to download it.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("PYTHONWARNINGS", "ignore:resource_tracker:UserWarning")

import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODEL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "day-10-clip-retrieval"
    / "pretrained"
    / "openai-clip-vit-base-patch32"
)
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_ACCEPT_THRESHOLD = 28.0


def default_model_reference() -> str:
    if DEFAULT_LOCAL_MODEL_DIR.exists():
        return str(DEFAULT_LOCAL_MODEL_DIR)
    return DEFAULT_MODEL_NAME


def load_transformers_classes() -> tuple[Any, Any]:
    try:
        from transformers import AutoProcessor, CLIPModel
    except ModuleNotFoundError:
        print(
            "The Python package 'transformers' is missing.\n"
            "Install project dependencies, then run this demo again."
        )
        raise SystemExit(1)
    return AutoProcessor, CLIPModel


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def draw_red_square() -> Image.Image:
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((52, 52, 172, 172), fill=(220, 30, 30), outline=(120, 0, 0), width=4)
    return image


def draw_blue_circle() -> Image.Image:
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((48, 48, 176, 176), fill=(40, 90, 220), outline=(10, 30, 130), width=4)
    return image


def draw_green_triangle() -> Image.Image:
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.polygon([(112, 42), (42, 176), (182, 176)], fill=(40, 170, 80), outline=(0, 90, 35))
    draw.line([(112, 42), (42, 176), (182, 176), (112, 42)], fill=(0, 90, 35), width=4)
    return image


def draw_noodle_bowl() -> Image.Image:
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((42, 114, 182, 188), fill=(230, 230, 235), outline=(90, 90, 100), width=4)
    draw.arc((54, 88, 170, 170), start=10, end=170, fill=(210, 150, 40), width=4)
    draw.arc((62, 94, 162, 166), start=10, end=170, fill=(210, 150, 40), width=4)
    draw.arc((70, 100, 154, 162), start=10, end=170, fill=(210, 150, 40), width=4)
    draw.line((60, 98, 178, 72), fill=(120, 80, 40), width=4)
    draw.line((66, 108, 184, 82), fill=(120, 80, 40), width=4)
    return image


def create_demo_gallery() -> tuple[list[str], list[Image.Image]]:
    names = [
        "generated red square",
        "generated blue circle",
        "generated green triangle",
        "generated noodle bowl drawing",
    ]
    images = [
        draw_red_square(),
        draw_blue_circle(),
        draw_green_triangle(),
        draw_noodle_bowl(),
    ]
    return names, images


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class ClipSearchEngine:
    def __init__(
        self,
        *,
        model_reference: str,
        device: torch.device,
        local_files_only: bool,
    ) -> None:
        AutoProcessor, CLIPModel = load_transformers_classes()
        self.device = device
        self.local_files_only = local_files_only
        self.model_reference = model_reference
        self.gallery_names, self.gallery_images = create_demo_gallery()
        self.gallery_data_urls = [image_to_data_url(image) for image in self.gallery_images]
        self.lock = threading.Lock()

        print(f"Loading CLIP model: {model_reference}")
        print(f"device = {device}")
        print(f"local_files_only = {local_files_only}")
        self.processor = AutoProcessor.from_pretrained(
            model_reference,
            local_files_only=local_files_only,
        )
        self.model = CLIPModel.from_pretrained(
            model_reference,
            local_files_only=local_files_only,
        )
        self.model.to(device)
        self.model.eval()

    def gallery_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "name": name,
                "image_data_url": data_url,
            }
            for index, (name, data_url) in enumerate(zip(self.gallery_names, self.gallery_data_urls))
        ]

    def search(self, query: str, *, top_k: int, accept_threshold: float) -> dict[str, Any]:
        top_k = max(1, min(top_k, len(self.gallery_images)))
        encoded = self.processor(
            text=[query],
            images=self.gallery_images,
            return_tensors="pt",
            padding=True,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}

        with self.lock, torch.inference_mode():
            outputs = self.model(**encoded)

        scores = outputs.logits_per_text.detach().cpu()[0]
        values, indices = torch.topk(scores, k=top_k, dim=0)
        top1_score = values[0].item()
        accepted = top1_score >= accept_threshold
        results = []
        for rank, (score, gallery_index) in enumerate(
            zip(values.tolist(), indices.tolist(), strict=True),
            start=1,
        ):
            results.append(
                {
                    "rank": rank,
                    "index": gallery_index,
                    "name": self.gallery_names[gallery_index],
                    "score": round(float(score), 4),
                    "image_data_url": self.gallery_data_urls[gallery_index],
                }
            )

        return {
            "query": query,
            "top_k": top_k,
            "accept_threshold": accept_threshold,
            "accepted": accepted,
            "decision": "accept" if accepted else "reject",
            "top1_score": round(float(top1_score), 4),
            "results": results,
            "shapes": {
                "pixel_values": list(encoded["pixel_values"].shape),
                "input_ids": list(encoded["input_ids"].shape),
                "logits_per_text": [1, len(self.gallery_images)],
            },
        }


def send_json(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(engine: ClipSearchEngine) -> type[BaseHTTPRequestHandler]:
    index_path = Path(__file__).with_name("index.html")

    class ClipSearchHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self.serve_index()
                return
            if parsed.path == "/api/gallery":
                send_json(
                    self,
                    {
                        "gallery": engine.gallery_payload(),
                        "model": engine.model_reference,
                        "device": str(engine.device),
                        "default_accept_threshold": DEFAULT_ACCEPT_THRESHOLD,
                    },
                )
                return
            if parsed.path == "/api/search":
                self.serve_search(parsed.query)
                return
            send_json(self, {"error": "not found"}, status=404)

        def serve_index(self) -> None:
            body = index_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def serve_search(self, query_string: str) -> None:
            params = parse_qs(query_string)
            query = params.get("q", [""])[0].strip()
            if not query:
                send_json(self, {"error": "query is required"}, status=400)
                return
            try:
                top_k = int(params.get("top_k", ["3"])[0])
                accept_threshold = float(
                    params.get("threshold", [str(DEFAULT_ACCEPT_THRESHOLD)])[0]
                )
            except ValueError:
                send_json(self, {"error": "top_k and threshold must be numeric"}, status=400)
                return

            try:
                payload = engine.search(query, top_k=top_k, accept_threshold=accept_threshold)
            except Exception as exc:  # pragma: no cover - demo server diagnostics.
                send_json(self, {"error": str(exc)}, status=500)
                return
            send_json(self, payload)

    return ClipSearchHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 10 minimal CLIP image-search web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--model", default=default_model_reference())
    parser.add_argument("--device", default="cpu", help="Use cpu by default for stable local demos.")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow model download if the CLIP model is not cached locally.",
    )
    return parser.parse_args()


def ensure_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            print(
                f"Port {host}:{port} is already in use.\n"
                f"Open the existing demo at http://{host}:{port}, or run with another port:\n"
                f"  .venv/bin/python demos/clip-image-search/app.py --port {port + 1}"
            )
            raise SystemExit(1)


def main() -> None:
    args = parse_args()
    ensure_port_available(args.host, args.port)
    engine = ClipSearchEngine(
        model_reference=args.model,
        device=choose_device(args.device),
        local_files_only=not args.allow_download,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"CLIP image search demo running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
