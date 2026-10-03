"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện bạn phải giữ:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar

Kiểm tra tự viết (focal gamma=0 ≡ CE, eps=0 ≡ CE, CutMix đúng diện tích...): test_losses.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

LOSS_CHOICES = ("ce", "ls", "focal", "ce_weighted")


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    kw: smoothing (ls, mặc định 0.1), gamma và alpha (focal), weight (ce_weighted, bắt buộc).
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted cần weight=class_weights(số ảnh mỗi lớp của train, beta)")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(kw["weight"], dtype=torch.float32))
    raise ValueError(f"loss={kind!r} không hợp lệ, chọn một trong {LOSS_CHOICES}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Tự cài đặt (không dùng CrossEntropyLoss(label_smoothing=...)); eps = 0 cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải nằm trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        uniform = -logp.mean(dim=1)          # CE với phân phối đều 1/K
        return ((1.0 - self.smoothing) * nll + self.smoothing * uniform).mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    alpha: None hoặc vector trọng số theo lớp. Lấy trung bình cộng trên batch.
    gamma = 0 và alpha = None cho đúng cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp_t = F.log_softmax(logits.float(), dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -((1.0 - logp_t.exp()) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = self.alpha[target] * loss
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN (không dùng val hay test).

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c)
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555)
    Cả hai đều chuẩn hoá tổng trọng số về số lớp (trung bình 1).
    """
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if (counts <= 0).any():
        raise ValueError("mọi lớp phải có ít nhất một ảnh train")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta phải nằm trong [0, 1)")
    w = 1.0 / counts if beta == 0 else (1.0 - beta) / (1.0 - beta ** counts)
    return (w * len(w) / w.sum()).float()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn; không sửa `x` tại chỗ.

    - lam ~ Beta(alpha, alpha), một giá trị cho cả batch; cặp ảnh lấy theo hoán vị ngẫu nhiên
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, rồi điều chỉnh lam theo
      DIỆN TÍCH THỰC của hộp sau khi cắt ra ngoài biên (slide trang 48)
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm], lam là float
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)

    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        ratio = math.sqrt(1.0 - lam)         # cạnh hộp tỉ lệ sqrt(1 - lam) để diện tích = 1 - lam
        cut_h, cut_w = int(h * ratio), int(w * ratio)
        cy, cx = int(torch.randint(h, (1,))), int(torch.randint(w, (1,)))
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
        x_mix = x.clone()
        x_mix[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    else:
        raise ValueError(f"mode={mode!r} không hợp lệ, chọn 'mixup' hoặc 'cutmix'")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    Lưu ý: accuracy trên batch đã trộn không còn nghĩa bình thường; đánh giá bằng val.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
