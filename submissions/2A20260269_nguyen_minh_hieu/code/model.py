"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float

Thêm so với khung:
    weights_tag(model)  -> str   tag trọng số timm thực sự được tải (ghi vào results.xlsx)
    train_mode(model)   -> None  thay cho model.train() trong train loop: giữ backbone đóng băng ở eval
"""
from __future__ import annotations

import timm
import torch
from torch import nn

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    # tag mặc định của timm là in12k_ft_in1k (tiền huấn luyện trên ImageNet-12k); chọn bản chỉ ImageNet-1k
    # để mọi backbone cùng nguồn dữ liệu tiền huấn luyện
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}
INIT_CHOICES = ("scratch", "frozen", "finetune")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ

    timm tự thay head mới 9 lớp (khởi tạo ngẫu nhiên). Tag trọng số đã tải: weights_tag(model).
    """
    if init not in INIT_CHOICES:
        raise ValueError(f"init={init!r} không hợp lệ, chọn một trong {INIT_CHOICES}")
    name = SUGGESTED_BACKBONES.get(name, name)
    pretrained = pretrained and init != "scratch"
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    model.pretrained_loaded = pretrained
    model.frozen_backbone = False
    if init == "frozen":
        freeze_backbone(model)
    return model


def weights_tag(model) -> str:
    """Tên đầy đủ `kiến_trúc.tag` của trọng số tiền huấn luyện, hoặc `kiến_trúc (scratch)`."""
    cfg = model.pretrained_cfg
    arch = cfg.get("architecture", type(model).__name__)
    tag = cfg.get("tag")
    return f"{arch}.{tag}" if getattr(model, "pretrained_loaded", False) and tag else f"{arch} (scratch)"


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head (model.get_classifier()).

    Backbone đóng băng thì BatchNorm cũng phải ở eval, nếu không running_mean/var vẫn bị cập nhật
    dù requires_grad=False. Vì vậy train loop gọi train_mode(model) thay cho model.train().
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    model.frozen_backbone = True


def train_mode(model) -> None:
    """model.train(), riêng model có backbone đóng băng thì chỉ head ở chế độ train."""
    if getattr(model, "frozen_backbone", False):
        model.eval()
        model.get_classifier().train()
    else:
        model.train()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
      (kèm các tham số timm khai báo trong model.no_weight_decay(): pos_embed, cls_token, bảng bias vị trí)
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay

    Bỏ qua tham số requires_grad == False và bỏ nhóm rỗng.
    """
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()

    decay, no_decay, head = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head.append(p)
        elif p.ndim <= 1 or name in skip or name.rsplit(".", 1)[-1] in skip:
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"name": "backbone", "params": decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"name": "backbone_no_decay", "params": no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"name": "head", "params": head, "lr": lr_head, "weight_decay": weight_decay},
    ]
    return [g for g in groups if g["params"]]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x).

    Công cụ: tự đếm bằng forward hook, không dùng thư viện ngoài. Đếm phép nhân-cộng của
    Conv2d, Linear và hai phép nhân ma trận trong self-attention (q·kᵀ và attn·v);
    bỏ qua norm, hàm kích hoạt, pooling và bias. Có thể lệch vài phần trăm so với fvcore/ptflops.
    """
    macs = 0

    def conv_hook(m, inputs, out):
        nonlocal macs
        macs += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m, inputs, out):
        nonlocal macs
        macs += out.numel() * m.in_features

    def attn_hook(m, inputs, out):
        nonlocal macs
        x = inputs[0]                       # (số cửa sổ, số token, chiều): tính trên mọi head gộp lại
        n_tokens = x.shape[1:-1].numel()
        macs += 2 * x.shape[0] * n_tokens * n_tokens * x.shape[-1]

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif hasattr(m, "qkv") and hasattr(m, "num_heads"):
            handles.append(m.register_forward_hook(attn_hook))

    modes = [(m, m.training) for m in model.modules()]
    p = next(model.parameters())
    model.eval()
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, img_size, img_size, device=p.device, dtype=p.dtype))
    finally:
        for h in handles:
            h.remove()
        for m, training in modes:
            m.training = training
    return macs / 1e9
