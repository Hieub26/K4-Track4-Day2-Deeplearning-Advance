"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers, seed)

Lựa chọn của bài này:
    - Val/test: ảnh gốc 256x256 -> CenterCrop(img_size) khi img_size < 256, giữ nguyên khi = 256,
      Resize(img_size) khi > 256 (chỉ dùng cho thí nghiệm dò độ phân giải kiểm tra).
    - Lật dọc ("vflip") là một giá trị `aug` riêng, không nằm trong "basic": ảnh chụp từ trên xuống
      nên lật dọc hợp lệ về ngữ nghĩa, nhưng công thức nền chỉ dùng crop + lật ngang (GUIDE 1.4).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509
AUG_CHOICES = ("basic", "vflip", "color", "trivial", "randaug", "mildcrop")


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Trả về ba DataFrame nguyên bản (cột `Filename, Label`), không sửa, lọc hay chia lại.
    """
    labels_dir = Path(labels_dir)
    return tuple(pd.read_csv(labels_dir / f"{name}_subset{fold}.csv") for name in ("train", "val", "test"))


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    Dừng ngay (AssertionError) nếu: trùng tên file trong một tập, giao hai tập khác rỗng,
    hợp ba tập khác 17.509 ảnh, nhãn ngoài 0..8, hoặc có file không tồn tại trong `images_dir`.
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    names = {k: set(df["Filename"]) for k, df in splits.items()}

    n = {k: len(df) for k, df in splits.items()}
    total = sum(n.values())
    per_class = {k: df["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).tolist()
                 for k, df in splits.items()}
    overlap = {f"{a}&{b}": len(names[a] & names[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    union = len(names["train"] | names["val"] | names["test"])
    on_disk = {p.name for p in Path(images_dir).iterdir()}
    missing = sorted(set().union(*names.values()) - on_disk)

    table = pd.DataFrame(per_class, index=CLASS_NAMES)
    table.loc["TỔNG"] = table.sum()
    print(table.to_string())
    print("tỉ lệ:", {k: f"{100 * v / total:.2f}%" for k, v in n.items()})
    print("giao:", overlap, "| hợp:", union, "| file thiếu:", len(missing))

    for k, df in splits.items():
        assert len(names[k]) == len(df), f"{k}: có tên file trùng lặp trong cùng một tập"
        assert df["Label"].between(0, NUM_CLASSES - 1).all(), f"{k}: nhãn ngoài 0..{NUM_CLASSES - 1}"
    assert not any(overlap.values()), f"các tập giao nhau: {overlap}"
    assert union == TOTAL_IMAGES, f"hợp ba tập = {union}, kỳ vọng {TOTAL_IMAGES}"
    assert not missing, f"{len(missing)} file không có trong {images_dir}, ví dụ {missing[:3]}"

    return {"n": n, "ratio": {k: v / total for k, v in n.items()}, "per_class": per_class,
            "overlap": overlap, "union": union, "missing": len(missing)}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` (chỉ có tác dụng khi train): basic | vflip | color | trivial | randaug | mildcrop.

    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.
    Train: RandomResizedCrop(img_size) + lật ngang (+ phép của `aug`) + ToTensor + Normalize.
    "mildcrop" đổi khoảng diện tích crop từ mặc định (0.08, 1) thành (0.35, 1): crop ít khi cắt mất cây cỏ.
    Val/test: không ngẫu nhiên; xem quy ước crop/resize ở docstring đầu file.
    """
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        if img_size < 256:
            return T.Compose([T.CenterCrop(img_size), *norm])
        if img_size > 256:
            return T.Compose([T.Resize(img_size), *norm])
        return T.Compose(norm)

    if aug not in AUG_CHOICES:
        raise ValueError(f"aug={aug!r} không hợp lệ, chọn một trong {AUG_CHOICES}")
    extra = {
        "basic": [],
        "vflip": [T.RandomVerticalFlip()],
        "color": [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05)],
        "trivial": [T.TrivialAugmentWide()],
        "randaug": [T.RandAugment(num_ops=2, magnitude=9)],
        "mildcrop": [],
    }[aug]
    scale = (0.35, 1.0) if aug == "mildcrop" else (0.08, 1.0)
    return T.Compose([T.RandomResizedCrop(img_size, scale=scale), T.RandomHorizontalFlip(), *extra, *norm])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.filenames = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        name = self.filenames[i]
        with Image.open(self.images_dir / name) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[i], name


def _seed_worker(worker_id: int) -> None:
    # torch đã đặt seed riêng cho từng worker từ generator của DataLoader; đồng bộ random và numpy theo nó
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2, seed: int = 0):
    """Tạo DataLoader.

    train=True: shuffle (hoặc sampler="balanced": WeightedRandomSampler, trọng số 1/(số ảnh của lớp),
    lấy mẫu có hoàn lại, mỗi epoch vẫn len(df) mẫu), drop_last=True.
    train=False: không shuffle, giữ thứ tự df để ghép logit với Filename.
    `seed` chỉ đổi thứ tự batch và augmentation ngẫu nhiên, không đổi cách chia (S5).
    """
    dataset = DeepWeedsDataset(df, images_dir, transform)
    generator = torch.Generator().manual_seed(seed)

    weighted = None
    if sampler == "balanced":
        if not train:
            raise ValueError("sampler chỉ dùng khi train")
        labels = np.asarray(dataset.labels)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        weighted = WeightedRandomSampler(torch.as_tensor(1.0 / counts[labels], dtype=torch.double),
                                         num_samples=len(dataset), replacement=True, generator=generator)
    elif sampler is not None:
        raise ValueError(f"sampler={sampler!r} không hợp lệ, chọn None hoặc 'balanced'")

    return DataLoader(dataset, batch_size=batch_size, shuffle=train and weighted is None, sampler=weighted,
                      drop_last=train, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0, worker_init_fn=_seed_worker, generator=generator)
