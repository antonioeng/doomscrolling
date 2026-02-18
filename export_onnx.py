"""
export_onnx.py — Export YOLOv8n to ONNX for NPU acceleration

Usage:
    python export_onnx.py                           # defaults
    python export_onnx.py --model yolov8n.pt --out vision/yolov8n.onnx --simplify

This exports the Ultralytics YOLOv8 model to ONNX format, which can then
be loaded by onnxruntime with DmlExecutionProvider (DirectML → NPU/GPU)
or TensorrtExecutionProvider.

After exporting:
  1. Set "use_onnx": true in config.json
  2. Set "onnx_model_path" to the exported .onnx path
  3. Set "npu.enabled": true and choose the correct execution_provider
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export_onnx")


def export(model_path: str, output_path: str, simplify: bool, opset: int) -> None:
    from ultralytics import YOLO

    logger.info("Loading model: %s", model_path)
    model = YOLO(model_path)

    logger.info("Exporting to ONNX (opset=%d, simplify=%s)...", opset, simplify)
    model.export(
        format="onnx",
        opset=opset,
        simplify=simplify,
        dynamic=False,
        imgsz=640,
    )

    # Ultralytics saves alongside the .pt by default; move if needed
    import shutil
    from pathlib import Path

    default_onnx = Path(model_path).with_suffix(".onnx")
    target = Path(output_path)

    if default_onnx.exists() and default_onnx != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(default_onnx), str(target))
        logger.info("Moved → %s", target)
    elif target.exists():
        logger.info("ONNX file ready: %s", target)
    else:
        logger.warning("Export produced no file at expected location.")

    logger.info("Done. To use NPU acceleration:")
    logger.info("  1. pip install onnxruntime-directml")
    logger.info('  2. Set "use_onnx": true  +  "npu.enabled": true  in config.json')


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument("--model", default="yolov8n.pt", help="Input .pt model")
    parser.add_argument("--out", default="vision/yolov8n.onnx", help="Output .onnx path")
    parser.add_argument("--simplify", action="store_true", help="ONNX simplifier")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    export(args.model, args.out, args.simplify, args.opset)


if __name__ == "__main__":
    main()
