"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)

Thêm so với khung:
    load_checkpoint(backbone, ckpt_path, device)     -> model đã nạp best.pt của một lần chạy
    predict_views(model, loader, device, views)      -> (filenames, y_true, [logits của từng view])
                                                        một lượt đọc dữ liệu cho mọi view của TTA
    fuse_error(model, fused, img_size)               -> sai số lớn nhất giữa trước và sau khi gộp BN
    uniform_soup(state_dicts)                        -> state_dict trung bình (model soup)

Giới hạn: views_multiscale và dò độ phân giải chỉ dùng được với CNN (có global pooling). DeiT/Swin của
timm cố định kích thước đầu vào 224 nên sẽ báo lỗi ở kích thước khác. Gộp BN không áp dụng cho
DeiT, Swin, ConvNeXt (dùng LayerNorm): fuse_conv_bn trả về bản sao không đổi và in ra 0 cặp đã gộp.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import model as model_lib


def load_checkpoint(backbone: str, ckpt_path, device, num_classes: int = 9):
    """Dựng lại kiến trúc (không tải trọng số tiền huấn luyện) rồi nạp checkpoint tốt nhất của một lần chạy."""
    net = model_lib.build_model(backbone, pretrained=False, num_classes=num_classes)
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return net.to(device).eval()


def predict_views(model, loader, device, views, amp: bool = True):
    """Chạy model trên loader với từng view trong `views` (list hàm batch -> batch).

    Trả về (filenames, y_true[N], [logits[N, 9] của từng view]) theo đúng thứ tự file của loader.
    """
    model.eval()
    filenames, ys, outs = [], [], [[] for _ in views]
    with torch.inference_mode():
        for x, y, names in loader:
            x = x.to(device, non_blocking=True)
            for k, view in enumerate(views):
                with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                    logits = model(view(x))
                outs[k].append(logits.float().cpu())
            ys.append(y)
            filenames.extend(names)
    return filenames, torch.cat(ys).numpy(), [torch.cat(o).numpy() for o in outs]


def predict_logits(model, loader, device, view=None, amp: bool = True):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    """
    filenames, y_true, (logits,) = predict_views(model, loader, device, [view or view_identity], amp)
    return filenames, y_true, logits


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def view_vflip(x):
    """Lật dọc batch (N, C, H, W)."""
    return torch.flip(x, dims=[-2])


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật ngang. Trả về list các batch.

    Dùng với loader đánh giá ở 256 (ảnh gốc) và crop=224.
    """
    h, w = x.shape[-2:]
    if crop > min(h, w):
        raise ValueError(f"crop={crop} lớn hơn ảnh {h}x{w}")
    top, left = (h - crop) // 2, (w - crop) // 2
    corners = [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop), (top, left)]
    crops = [x[..., t:t + crop, l:l + crop] for t, l in corners]
    return crops + [view_hflip(c) for c in crops] if flip else crops


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch (chỉ cho CNN, xem đầu file)."""
    return [x if s == x.shape[-1] else F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False,
                                                     antialias=True)
            for s in sizes]


def _softmax(logits, T: float = 1.0) -> np.ndarray:
    return torch.as_tensor(np.asarray(logits), dtype=torch.float64).div(T).softmax(dim=-1).numpy()


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62). Trả về xác suất (N, 9).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    """
    stacked = np.stack([np.asarray(l, dtype=np.float64) for l in logits_per_view])
    if space == "prob":
        return _softmax(stacked).mean(axis=0)
    if space == "logit":
        return _softmax(stacked.mean(axis=0))
    raise ValueError(f"space={space!r} không hợp lệ, chọn 'prob' hoặc 'logit'")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed).

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file
    (người gọi phải kiểm tra danh sách tên file trùng nhau).
    """
    probs = [np.asarray(p, dtype=np.float64) for p in list_of_probs]
    if len({p.shape for p in probs}) != 1:
        raise ValueError(f"các mô hình khác số ảnh hoặc số lớp: {[p.shape for p in probs]}")
    return np.mean(probs, axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    LBFGS trên log T (để T luôn dương). Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    logits = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    labels = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T)."""
    return _softmax(logits, T)


def _fuse_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    scale = torch.rsqrt(bn.running_var + bn.eps)
    if bn.weight is not None:
        scale = scale * bn.weight
    bias = torch.zeros_like(bn.running_mean) if conv.bias is None else conv.bias
    bias = (bias - bn.running_mean) * scale
    if bn.bias is not None:
        bias = bias + bn.bias

    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode)
    fused = fused.to(device=conv.weight.device, dtype=conv.weight.dtype)
    fused.weight.data.copy_(conv.weight * scale.view(-1, 1, 1, 1))
    fused.bias.data.copy_(bias)
    return fused


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Trả về BẢN SAO đã gộp (model gốc giữ nguyên) ở chế độ eval, và in số cặp đã gộp.
    Cặp (Conv2d, BatchNorm2d) được tìm theo thứ tự khai báo liền kề trong cùng một module cha.
    BN kèm hàm kích hoạt của timm (BatchNormAct2d) được thay bằng chính phần drop + kích hoạt của nó.
    Luôn kiểm tra lại bằng fuse_error(model, fused): sai số phải cỡ 1e-5 trở xuống.
    """
    fused = copy.deepcopy(model).eval()
    n_pairs = 0
    for parent in fused.modules():
        children = list(parent.named_children())
        for (conv_name, conv), (bn_name, bn) in zip(children, children[1:]):
            if not (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)):
                continue
            if type(conv) is not nn.Conv2d and type(conv).forward is not nn.Conv2d.forward:
                continue    # conv tự định nghĩa forward (ví dụ padding động): không gộp
            if conv.out_channels != bn.num_features:
                continue
            setattr(parent, conv_name, _fuse_pair(conv, bn))
            rest = [m for m in (getattr(bn, "drop", None), getattr(bn, "act", None)) if m is not None]
            setattr(parent, bn_name, nn.Sequential(*rest) if rest else nn.Identity())
            n_pairs += 1
    print(f"fuse_conv_bn: đã gộp {n_pairs} cặp Conv2d + BatchNorm2d")
    return fused


def fuse_error(model, fused, img_size: int = 224, batch_size: int = 4) -> float:
    """Sai số tuyệt đối lớn nhất của logit giữa model gốc và model đã gộp BN, trên đầu vào ngẫu nhiên."""
    p = next(model.parameters())
    x = torch.randn(batch_size, 3, img_size, img_size, device=p.device, dtype=p.dtype,
                    generator=torch.Generator(p.device).manual_seed(0))
    was_training = [(m, m.training) for m in model.modules()]
    model.eval()
    with torch.inference_mode():
        err = float((model(x) - fused(x)).abs().max())
    for m, training in was_training:
        m.training = training
    return err


def uniform_soup(state_dicts):
    """Model soup đều: trung bình trọng số của nhiều checkpoint CÙNG kiến trúc (slide trang 67).

    Tham số và buffer kiểu số thực được lấy trung bình; buffer nguyên (num_batches_tracked) lấy của bản đầu.
    """
    soup = {}
    for key, first in state_dicts[0].items():
        if first.is_floating_point():
            soup[key] = torch.stack([sd[key].to(first.device) for sd in state_dicts]).mean(dim=0)
        else:
            soup[key] = first.clone()
    return soup
