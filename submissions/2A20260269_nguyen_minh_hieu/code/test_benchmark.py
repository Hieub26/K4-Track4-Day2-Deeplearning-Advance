"""Kiểm tra tự viết cho benchmark.py. Chạy trong thư mục code/, không cần GPU:
    python -m unittest test_benchmark -v
"""
import time
import unittest

import torch
from torch import nn

import benchmark


class TestBench(unittest.TestCase):
    def test_warmup_is_discarded_and_sync_wraps_each_measurement(self):
        calls = {"fn": 0, "sync": 0}

        def fn():
            calls["fn"] += 1
            time.sleep(0.05 if calls["fn"] <= 10 else 0.002)    # 10 lần đầu chậm: phải bị bỏ

        r = benchmark.bench(fn, warmup=10, iters=50, sync=lambda: calls.__setitem__("sync", calls["sync"] + 1))
        self.assertEqual(calls["fn"], 60)
        self.assertEqual(calls["sync"], 100)           # trước và sau mỗi lần đo
        self.assertEqual(r["n"], 50)
        self.assertLess(r["p99"], 40.0)                # không dính các lần warmup 50 ms
        self.assertGreaterEqual(r["p50"], 2.0)
        self.assertTrue(r["p50"] <= r["p95"] <= r["p99"])

    def test_rejects_too_few_runs(self):
        with self.assertRaises(ValueError):
            benchmark.bench(lambda: None, warmup=0, iters=100)
        with self.assertRaises(ValueError):
            benchmark.bench(lambda: None, warmup=10, iters=10)


class TestReports(unittest.TestCase):
    def setUp(self):
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 9))

    def test_latency_report_cpu(self):
        r = benchmark.latency_report(self.net, batch_size=2, img_size=32, device="cpu", iters=50, config="X")
        for key in ("gpu", "dtype", "batch", "img_size", "p50", "p95", "p99", "images_per_s", "torch", "config"):
            self.assertIn(key, r)
        self.assertEqual((r["gpu"], r["dtype"], r["batch"]), ("cpu", "fp32", 2))
        self.assertAlmostEqual(r["images_per_s"], 2 / (r["p50"] / 1000), places=6)

    def test_half_precision_needs_gpu(self):
        with self.assertRaises(ValueError):
            benchmark.latency_report(self.net, batch_size=1, img_size=32, dtype="fp16", device="cpu")

    def test_tta_latency_cpu(self):
        r = benchmark.tta_latency(self.net, 4, batch_size=1, img_size=64, device="cpu", iters=50, config="X")
        self.assertEqual((r["k_views"], r["config"]), (4, "X"))
        self.assertGreater(r["p50"], r["single_p50"])

    @unittest.skipUnless(torch.cuda.is_available(), "cần GPU")
    def test_fp16_does_not_change_callers_model(self):
        benchmark.latency_report(self.net, batch_size=1, img_size=32, dtype="fp16", iters=50)
        self.assertEqual(next(self.net.parameters()).dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
