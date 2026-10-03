"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không

Lựa chọn của bài này: KHÔNG tính tiền xử lý. Chỉ đo lượt forward của model trên một tensor ngẫu nhiên
đã nằm sẵn trên thiết bị (không đọc ảnh, không resize, không chép CPU -> GPU).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

DTYPES = ("fp32", "amp", "fp16")


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    Trả về {"p50", "p95", "p99", "mean", "n"}.
    """
    if warmup < 10 or iters < 50:
        raise ValueError("cần warmup >= 10 và iters >= 50 (GUIDE.md mục 4.1)")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    p50, p95, p99 = np.percentile(times, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "mean": float(np.mean(times)), "n": iters}


def _forward_fn(model, batch_size: int, img_size: int, dtype: str, device: str, k_views: int = 1):
    """Chuẩn bị model (bản sao khi fp16) và đầu vào; trả về (hàm chạy k_views lượt forward, hàm sync)."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype={dtype!r} không hợp lệ, chọn một trong {DTYPES}")
    dev = torch.device(device)
    if dtype != "fp32" and dev.type != "cuda":
        raise ValueError("amp và fp16 chỉ đo trên GPU")

    model = model.to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        model, x = copy.deepcopy(model).half(), x.half()   # bản sao: không đổi dtype model của người gọi

    def fn():
        with torch.inference_mode(), torch.autocast(dev.type, enabled=dtype == "amp"):
            for _ in range(k_views):
                model(x)

    return fn, (torch.cuda.synchronize if dev.type == "cuda" else None)


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, **labels) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast) | "fp16" (model.half()). `labels` (ví dụ config="F01", fused_bn=True)
    được chép vào kết quả. Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx.
    """
    fn, sync = _forward_fn(model, batch_size, img_size, dtype, device)
    r = bench(fn, warmup, iters, sync)
    return {**labels,
            "gpu": torch.cuda.get_device_name(0) if torch.device(device).type == "cuda" else "cpu",
            "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "mean": r["mean"], "n": r["n"],
            "images_per_s": batch_size / (r["p50"] / 1000.0), "torch": torch.__version__}


def pipeline_latency(parts, batch_size: int, dtype: str = "fp32", device: str = "cuda",
                     warmup: int = 10, iters: int = 100, **labels) -> dict:
    """Độ trễ của một quy trình suy luận gồm nhiều lượt forward nối tiếp, đo thật trong một lần bấm giờ.

    `parts` là list (model, img_size, số lượt): ví dụ TTA 5 crop = [(m, 224, 5)], ensemble hai mô hình =
    [(m1, 224, 1), (m2, 224, 1)], TTA lật ở 288 = [(m, 288, 2)]. Trả về dict như latency_report,
    với "img_size" và "forwards" mô tả các phần.
    """
    fns = [_forward_fn(m, batch_size, img, dtype, device, k) for m, img, k in parts]

    def fn():
        for f, _ in fns:
            f()

    r = bench(fn, warmup, iters, fns[0][1])
    return {**labels,
            "gpu": torch.cuda.get_device_name(0) if torch.device(device).type == "cuda" else "cpu",
            "dtype": dtype, "batch": batch_size, "img_size": "+".join(str(img) for _, img, _ in parts),
            "forwards": sum(k for _, _, k in parts),
            "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "mean": r["mean"], "n": r["n"],
            "images_per_s": batch_size / (r["p50"] / 1000.0), "torch": torch.__version__}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view (K lượt forward nối tiếp trên cùng batch), đo thật (slide trang 63).

    `kw` như latency_report. Kết quả có thêm "k_views", "single_p50" (một lượt) và
    "ratio_vs_k_single" = p50 của TTA / (K * p50 một lượt), kỳ vọng xấp xỉ 1.
    """
    labels = {k: kw.pop(k) for k in list(kw) if k not in
              ("batch_size", "img_size", "dtype", "device", "warmup", "iters")}
    single = latency_report(model, **kw)
    fn, sync = _forward_fn(model, kw["batch_size"], kw["img_size"], kw.get("dtype", "fp32"),
                           kw.get("device", "cuda"), k_views)
    r = bench(fn, kw.get("warmup", 10), kw.get("iters", 100), sync)
    return {**single, **labels, "k_views": k_views, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
            "mean": r["mean"], "images_per_s": kw["batch_size"] / (r["p50"] / 1000.0),
            "single_p50": single["p50"], "ratio_vs_k_single": r["p50"] / (k_views * single["p50"])}
