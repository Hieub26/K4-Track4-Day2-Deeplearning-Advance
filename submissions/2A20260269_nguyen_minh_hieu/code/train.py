"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm. train.py tự tìm eval.py ở các thư mục cha của nó.

Lựa chọn của bài này:
    - Lịch LR cập nhật theo BƯỚC (iteration): warmup tuyến tính rồi cosine về 0.
    - Loss val luôn là cross-entropy thường, để so sánh được giữa các thí nghiệm khác loss;
      loss train là loss đang tối ưu (kể cả khi Mixup/CutMix).
    - Có EMA thì đánh giá, chọn checkpoint và lưu checkpoint bằng trọng số EMA.
    - Tái lập: cố định seed cho random/numpy/torch và DataLoader; cudnn.benchmark bật để nhanh,
      nên hai lần chạy cùng seed trên GPU có thể lệch nhẹ ở chữ số cuối.
    - Chạy lại một cấu hình đã xong (đã có summary.json) thì bỏ qua phần huấn luyện; dùng overwrite=True
      để huấn luyện lại. Bật save_test_predictions cho một lần chạy đã xong sẽ nạp checkpoint tốt nhất
      và chỉ chạy test, nếu file dự đoán test chưa tồn tại.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import dataset
import losses
import model as model_lib

for _parent in Path(__file__).resolve().parents:
    if (_parent / "eval.py").exists():
        sys.path.insert(0, str(_parent))
        break
import eval as ev  # noqa: E402  (eval.py của repo gốc: save_predictions, compute_metrics)


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    desc: str = ""                    # mô tả ngắn cho tên ảnh curves/<exp_id>_<desc>.png (mặc định: backbone)
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | vflip | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    optimizer: str = "adamw"          # adamw | sgd (momentum 0.9, nesterov)
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    grad_clip: float | None = None
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curve_dir: str = "curves"         # ảnh biểu đồ training (nộp cùng bài)
    overwrite: bool = False           # True: huấn luyện lại dù lần chạy này đã có summary.json
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    """Ảnh biểu đồ: <curve_dir>/<exp_id>_<desc>.png; seed khác 0 thêm hậu tố _seed<k>."""
    seed = "" if cfg.seed == 0 else f"_seed{cfg.seed}"
    return Path(cfg.curve_dir) / f"{cfg.exp_id}_{cfg.desc or cfg.backbone}{seed}.png"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên: random, numpy, torch (CPU và CUDA).

    Worker của DataLoader được seed trong dataset.make_loader. cudnn.benchmark bật (xem docstring đầu file).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD) với 3 nhóm tham số (xem model.param_groups)."""
    groups = model_lib.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(f"optimizer={cfg.optimizer!r} không hợp lệ, chọn 'adamw' hoặc 'sgd'")


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0 (slide trang 55), cập nhật theo BƯỚC.

    Hệ số nhân chung cho LR của mọi nhóm: gọi scheduler.step() sau mỗi optimizer.step().
    """
    total = cfg.epochs * steps_per_epoch
    warmup = int(cfg.warmup_epochs * steps_per_epoch)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    Giữ một bản sao riêng của model (`self.module`, luôn ở eval) để đánh giá bằng trọng số EMA.
    Buffer (running_mean/var của BatchNorm) được chép thẳng từ model sau mỗi bước, không lấy trung bình.
    Những bước đầu dùng decay nhỏ hơn, d = min(decay, (1 + n) / (10 + n)), để EMA không bị kéo về
    trọng số khởi tạo khi số bước huấn luyện ít.
    """

    def __init__(self, model, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model) -> None:
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1.0 - d)
        for e, b in zip(self.module.buffers(), model.buffers()):
            e.copy_(b)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "lr" (nhóm đầu, cuối epoch), "lr_steps" (theo bước)}."""
    model_lib.train_mode(model)   # backbone đóng băng thì vẫn ở eval
    use_amp = scaler.is_enabled()
    total_loss, n_seen, lr_steps = 0.0, 0, []

    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if cfg.mix:
            x, targets = losses.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        with torch.autocast(device.type, enabled=use_amp):
            logits = model(x)
            loss = losses.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        lr_steps.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * x.size(0)
        n_seen += x.size(0)

    if not math.isfinite(total_loss):
        raise RuntimeError("loss train là NaN/inf: giảm LR hoặc tắt AMP")
    return {"train_loss": total_loss / n_seen, "lr": lr_steps[-1], "lr_steps": lr_steps}


def evaluate(model, loader, criterion, device, amp: bool = True):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file.
    """
    model.eval()
    filenames, ys, outs = [], [], []
    with torch.inference_mode():
        for x, y, names in loader:
            with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                logits = model(x.to(device, non_blocking=True))
            outs.append(logits.float().cpu())
            ys.append(y)
            filenames.extend(names)
        logits, y_true = torch.cat(outs), torch.cat(ys)
        loss = float(criterion(logits, y_true))
    return filenames, y_true.numpy(), logits.numpy(), loss


def softmax(logits: np.ndarray) -> np.ndarray:
    return torch.from_numpy(logits).double().softmax(dim=1).numpy()


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps: list[float] | None = None) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    Ba ô: loss train/val theo epoch; macro-F1 và top-1 val theo epoch (đánh dấu epoch tốt nhất);
    LR theo bước (nếu có `lr_steps`, không thì LR cuối mỗi epoch).
    """
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    best = h.loc[h["val_macro_f1"].idxmax()]          # idxmax lấy epoch sớm nhất khi hòa
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(h["epoch"], h["train_loss"], "o-", label="train")
    ax.plot(h["epoch"], h["val_loss"], "o-", label="val (CE)")
    ax.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax.legend()

    ax = axes[1]
    ax.plot(h["epoch"], h["val_macro_f1"], "o-", label="macro-F1 val")
    ax.plot(h["epoch"], h["val_top1"], "o-", label="top-1 val")
    ax.axvline(best["epoch"], color="gray", linestyle="--",
               label=f"tốt nhất: epoch {int(best['epoch'])}, F1 {best['val_macro_f1']:.4f}")
    ax.set(xlabel="epoch", ylabel="chỉ số trên val", title="Chỉ số val")
    ax.legend()

    ax = axes[2]
    if lr_steps:
        ax.plot(range(1, len(lr_steps) + 1), lr_steps)
        ax.set_xlabel("bước (iteration)")
    else:
        ax.plot(h["epoch"], h["lr"], "o-")
        ax.set_xlabel("epoch")
    ax.set(ylabel="LR (nhóm tham số đầu)", title="Lịch LR")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _criterion(cfg: Config, train_df) -> nn.Module:
    counts = train_df["Label"].value_counts().reindex(range(dataset.NUM_CLASSES), fill_value=0).tolist()
    weight = losses.class_weights(counts, cfg.class_weight_beta or 0.0) if cfg.loss == "ce_weighted" else None
    return losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing or 0.1, gamma=cfg.focal_gamma,
                                  weight=weight)


def _metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    return ev.compute_metrics(y_true, logits.argmax(1), softmax(logits))


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    Lưu trong run_dir(cfg): config.json, history.csv, lr_steps.npy, best.pt, val_logits.npy,
    summary.json (và test_logits.npy nếu chạy test); ngoài ra predictions/*.csv và curves/*.png.
    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    """
    out = run_dir(cfg)
    summary_path, ckpt_path = out / "summary.json", out / "best.pt"
    trained = summary_path.exists() and not cfg.overwrite
    need_test = cfg.save_test_predictions and (cfg.overwrite or not pred_path(cfg, "test").exists())
    if trained and not need_test:
        print(f"[{cfg.exp_id} seed{cfg.seed}] đã chạy xong, bỏ qua (overwrite=True để chạy lại)")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if trained and not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} không còn: đặt overwrite=True để huấn luyện lại rồi chạy test")

    set_seed(cfg.seed)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"

    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    eval_tf = dataset.build_transforms(False, cfg.img_size)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, eval_tf, cfg.batch_size, train=False,
                                     num_workers=cfg.num_workers, seed=cfg.seed)

    net = model_lib.build_model(cfg.backbone, drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    val_criterion = nn.CrossEntropyLoss()

    if trained:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        net.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        info = {"weights_tag": model_lib.weights_tag(net), "params_m": model_lib.count_params(net),
                "gmacs": model_lib.count_gmacs(net, cfg.img_size),
                "python": platform.python_version(), "torch": torch.__version__,
                "timm": model_lib.timm.__version__,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
        (out / "config.json").write_text(
            json.dumps({**dataclasses.asdict(cfg), **info}, indent=2, ensure_ascii=False), encoding="utf-8")

        train_loader = dataset.make_loader(
            train_df, cfg.images_dir, dataset.build_transforms(True, cfg.img_size, cfg.aug), cfg.batch_size,
            train=True, sampler=cfg.sampler, num_workers=cfg.num_workers, seed=cfg.seed)
        criterion = _criterion(cfg, train_df).to(device)
        optimizer = build_optimizer(net, cfg)
        scheduler = build_scheduler(optimizer, cfg, len(train_loader))
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        ema = EMA(net, cfg.ema_decay) if cfg.ema_decay else None
        eval_net = ema.module if ema else net

        history, lr_steps, best_f1 = [], [], -1.0
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            stats = train_one_epoch(net, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_time = time.perf_counter() - t0
            lr_steps += stats.pop("lr_steps")

            _, y_val, logits, val_loss = evaluate(eval_net, val_loader, val_criterion, device, use_amp)
            m = _metrics(y_val, logits)
            history.append({"epoch": epoch, **stats, "val_loss": val_loss, "val_macro_f1": m["macro_f1"],
                            "val_top1": m["top1"], "train_time_s": train_time})
            if m["macro_f1"] > best_f1:       # dấu > : hòa thì giữ epoch sớm hơn
                best_f1 = m["macro_f1"]
                torch.save(eval_net.state_dict(), ckpt_path)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            print(f"[{cfg.exp_id} seed{cfg.seed}] epoch {epoch:2d}/{cfg.epochs} "
                  f"train_loss {stats['train_loss']:.4f} val_loss {val_loss:.4f} "
                  f"val_macro_f1 {m['macro_f1']:.4f} val_top1 {m['top1']:.4f} ({train_time:.0f}s)", flush=True)

        np.save(out / "lr_steps.npy", np.asarray(lr_steps))
        plot_curves(history, curve_path(cfg), f"{cfg.exp_id} · {info['weights_tag']} · seed {cfg.seed}", lr_steps)

        net.load_state_dict(torch.load(ckpt_path, map_location=device))
        names, y_val, logits, val_loss = evaluate(net, val_loader, val_criterion, device, use_amp)
        np.save(out / "val_logits.npy", logits)
        ev.save_predictions(pred_path(cfg, "val"), names, y_val, softmax(logits))
        m = _metrics(y_val, logits)
        h = pd.DataFrame(history)
        summary = {"exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, **info,
                   "best_epoch": int(h.loc[h["val_macro_f1"].idxmax(), "epoch"]), "epochs": cfg.epochs,
                   "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_balanced_acc": m["balanced_acc"],
                   "val_ece": m["ece"], "val_loss": val_loss, "val_f1_per_class": m["f1"].tolist(),
                   "train_time_per_epoch_s": float(h["train_time_s"].mean()),
                   "curve": str(curve_path(cfg))}
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if need_test:   # chỉ ở Bước 4: test đúng MỘT lần cho mỗi seed, bằng checkpoint đã chọn trên val
        test_loader = dataset.make_loader(test_df, cfg.images_dir, eval_tf, cfg.batch_size, train=False,
                                          num_workers=cfg.num_workers, seed=cfg.seed)
        names, y_test, logits, _ = evaluate(net, test_loader, val_criterion, device, use_amp)
        np.save(out / "test_logits.npy", logits)
        ev.save_predictions(pred_path(cfg, "test"), names, y_test, softmax(logits))
        print(f"[{cfg.exp_id} seed{cfg.seed}] đã ghi {pred_path(cfg, 'test')}")

    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    types = {f.name: f.type for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"{pair!r}: phải có dạng KEY=VALUE")
        if key not in types:
            raise ValueError(f"{key!r} không phải field của Config; các field: {sorted(types)}")
        kind = types[key]
        if "None" in kind and value.lower() in ("none", "null", ""):
            out[key] = None
        elif kind.startswith("bool"):
            if value.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"{key}: cần true/false, nhận {value!r}")
            out[key] = value.lower() in ("true", "1")
        elif kind.startswith("int"):
            out[key] = int(value)
        elif kind.startswith("float"):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="ghi đè field của Config")
    args = parser.parse_args()
    summary = run(Config(**parse_overrides(args.set)))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
