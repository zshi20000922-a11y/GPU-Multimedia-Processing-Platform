#!/usr/bin/env python3
"""Export a YOLO model to dynamic-batch ONNX and a TensorRT FP16 plan."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True,
                        help="Local Ultralytics model weights (.pt)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--trtexec", default="trtexec",
                        help="Path to TensorRT trtexec executable")
    parser.add_argument("--input-name",
                        help="ONNX input tensor name used by trtexec profiles")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--workspace-mib", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not weights.is_file():
        raise SystemExit(f"Weights file not found: {weights}")
    if args.max_batch < 1 or args.image_size < 1 or args.workspace_mib < 1:
        raise SystemExit("Image size, max batch, and workspace must be positive")

    trtexec = shutil.which(args.trtexec)
    if trtexec is None:
        raise SystemExit(f"TensorRT trtexec was not found: {args.trtexec}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install the optional Python dependency: pip install ultralytics onnx") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    exported_path = Path(model.export(
        format="onnx",
        imgsz=args.image_size,
        dynamic=True,
        simplify=True,
        opset=12,
        half=False,
    )).resolve()
    if not exported_path.is_file():
        raise SystemExit(f"Ultralytics did not produce an ONNX file: {exported_path}")

    try:
        import onnx
    except ImportError as exc:
        raise SystemExit("Install the optional ONNX checker: pip install onnx") from exc

    graph = onnx.load(str(exported_path))
    onnx.checker.check_model(graph)
    if not graph.graph.input:
        raise SystemExit("Exported ONNX model has no input tensor")
    onnx_input = graph.graph.input[0].name
    input_name = args.input_name or onnx_input

    stem = weights.stem
    onnx_path = output_dir / f"{stem}.onnx"
    engine_path = output_dir / f"{stem}_fp16.engine"
    if exported_path != onnx_path:
        shutil.copy2(exported_path, onnx_path)

    shape = f"3x{args.image_size}x{args.image_size}"
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--minShapes={input_name}:1x{shape}",
        f"--optShapes={input_name}:1x{shape}",
        f"--maxShapes={input_name}:{args.max_batch}x{shape}",
        f"--workspace={args.workspace_mib}",
    ]
    print("Building TensorRT engine:", " ".join(command))
    subprocess.run(command, check=True)
    print(f"ONNX:   {onnx_path}")
    print(f"Engine: {engine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
