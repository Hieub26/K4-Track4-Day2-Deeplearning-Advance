# Lab Day 2 — DeepWeeds — 2A20260269 Nguyễn Minh Hiếu

## Notebook chạy lại

- Notebook: [`code/lab_day2.ipynb`](code/lab_day2.ipynb), chạy trên Kaggle (GPU T4, bật Internet) bằng *Save & Run All*.
- Link Kaggle: _(chưa có — điền sau khi chạy xong)_

Notebook tự `git clone` repo này, tải `images.zip` từ Zenodo (kiểm tra MD5) và các CSV fold 0 từ GitHub của tác giả.

## Thứ tự chạy

Chạy các ô của notebook từ trên xuống:

| Bước | Nội dung | exp_id |
|---|---|---|
| 0 | Kiểm tra chia dữ liệu (README gốc mục 2.1), EDA, kiểm tra pipeline (loss ban đầu, overfit một batch, ảnh sau augmentation) | — |
| 1 | 7 backbone, cùng công thức nền, seed 0 | `B01`–`B07` |
| 2 | Công thức nền 3 seed; mỗi ablation khác nền một yếu tố (trục A–G), seed 0; một tổ hợp | `T00`, `T01`–`T16`, `T20` |
| 3 | Các phương pháp suy luận trên val và độ trễ p50/p95/p99 | `I00`–`I08` |
| 4 | Chung kết 3 seed, test một lần mỗi seed, `eval.py score` và `eval.py grade` | `F01`, `F01_uncal`, `F01rt`, `T00` |
| 5 | `results.xlsx`, kiểm tra ảnh biểu đồ, gói `submission_outputs.zip` | — |

Mọi lựa chọn (backbone đi tiếp, tổ hợp công thức, phương pháp suy luận) lấy theo số liệu **val**; test chỉ được đọc trong
`experiments.final_predictions()` và `train.run(..., save_test_predictions=True)` ở Bước 4.

Chạy một thí nghiệm riêng từ dòng lệnh (trong thư mục có `data/`):

```bash
python submissions/2A20260269_nguyen_minh_hieu/code/train.py --set exp_id=B01 backbone=resnet50 seed=0
```

## Seed

- Sàng backbone và ablation: seed 0.
- Mốc `T00` và chung kết `F01`: seed 0, 1, 2.
- Seed chỉ đổi khởi tạo head, thứ tự batch và augmentation ngẫu nhiên; cách chia luôn là fold 0 chia sẵn.

## Code

| File | Nội dung |
|---|---|
| `code/dataset.py` | Đọc CSV, kiểm tra chia dữ liệu, transform/augmentation, `Dataset`, `DataLoader` |
| `code/model.py` | Backbone `timm`, đóng băng, 3 nhóm tham số, đếm tham số và GMAC |
| `code/losses.py` | Label smoothing, focal loss, trọng số lớp, Mixup/CutMix |
| `code/train.py` | `run(Config)` dùng chung cho mọi thí nghiệm: AMP, warmup + cosine, EMA, chọn checkpoint theo macro-F1 val |
| `code/inference.py` | TTA, gộp xác suất/logit, ensemble, temperature scaling, gộp BatchNorm |
| `code/benchmark.py` | Đo độ trễ p50/p95/p99 (warmup, `cuda.synchronize`, ≥ 50 lần đo) |
| `code/experiments.py` | Danh sách thí nghiệm, chọn cấu hình trên val, nghiên cứu suy luận, bảng và `results.xlsx` |
| `code/test_*.py` | Kiểm tra tự viết: focal γ=0 ≡ CE, CutMix, gộp BN, temperature scaling, đo độ trễ, bảng |

`eval.py` là bản gốc của repo, không sửa.

Kiểm tra tự viết (không cần GPU), chạy trong `code/`:

```bash
python -m unittest test_losses test_inference test_benchmark test_experiments
```

Trên Windows thêm `PYTHONUTF8=1` trước lệnh `python` (kể cả khi chạy `eval.py`), vì console mặc định không in được tiếng Việt.

## Phiên bản thư viện

Phiên bản dùng khi chạy thật được notebook in ở ô đầu và ghi vào `choices.json` cùng `config.json` của từng lần chạy.
_(Điền phiên bản Kaggle sau khi chạy xong.)_

Môi trường dùng để viết và kiểm tra code: Python 3.13.9, torch 2.6.0+cu124, torchvision 0.21.0, timm 1.0.30.

## Checkpoint

Không commit checkpoint. _(Điền link nếu cần chia sẻ.)_
