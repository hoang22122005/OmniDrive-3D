# OmniShield-X

Mã nguồn và dữ liệu mẫu nằm trong workspace. Huấn luyện dùng CUDA trong Kaggle. Bản replay mô phỏng trong [mvp](mvp/README.md) chạy độc lập.

| Đường dẫn | Nội dung |
|---|---|
| [kaggle/OmniShield.ipynb](kaggle/OmniShield.ipynb) | Notebook chuẩn bị dữ liệu, smoke test, huấn luyện, resume và đánh giá |
| [kaggle/jobs.json](kaggle/jobs.json) | Bảy công việc với run ID độc lập |
| `configs/model`, `configs/rig`, `configs/train` | Hình ảnh, lưới, rig và cấu hình huấn luyện |
| `omnishield/data`, `omnishield/geometry` | Manifest, adapter PHENIKAA, nhãn, phép chiếu và LUT |
| `omnishield/models`, `omnishield/engine`, `omnishield/eval` | DINOv3, lifting, các head voxel, loss, checkpoint, dự đoán và metric |
| `data/raw`, `data/prepared`, `weights`, `runs` | Quy ước gắn dữ liệu, trọng số và nhận đầu ra |
| [docs/kaggle_training.md](docs/kaggle_training.md) | Thứ tự chạy và hợp đồng dữ liệu |
| [docs/datasets.md](docs/datasets.md) | Danh mục bộ dữ liệu train Pro và đối chiếu |
| [docs/omnishield_x_implementation_spec.md](docs/omnishield_x_implementation_spec.md) | Thiết kế nghiên cứu tổng thể |

Đính kèm `dist/omnishield-src.zip` và mở notebook trên Kaggle. Tuyến MVP thật dùng `prepare-mvp → mvp-real → replay-mvp`. Profile này chuẩn bị hai clip PHENIKAA công khai, train DINOv3 và xuất replay RGB-only trên clip validation. Dùng source DINOv3 và trọng số được cấp quyền. LiDAR và nhãn dùng để huấn luyện và đánh giá. [Hướng dẫn MVP trên dữ liệu thật](docs/phenikaa_mvp.md)

Tạo lại gói khi thay code, từ gốc workspace:

```powershell
python -m scripts.package_kaggle
```

Lệnh trên chỉ đóng gói source và config. Các tệp dữ liệu, trọng số và checkpoint đi qua input/output riêng của Kaggle.

## Đo bản Edge trên dữ liệu thật

Từ gốc workspace, dùng checkpoint `mvp_v5_best.pt` và clip validation PHENIKAA để chạy bản tham chiếu trên CPU. Lệnh `export` cần nhóm phụ thuộc `export` và kiểm tra đầu ra ONNX Runtime trên khung ghi được, từng ca mất một camera và ca mất toàn bộ camera.

```powershell
$edgeArgs = @(
    '--raw', 'data/raw/phenikaa-mini-mvp',
    '--prepared', 'data/prepared/phenikaa-mini-mvp',
    '--checkpoint', 'weights/mvp_v5_best.pt',
    '--dino-source', '.cache/dinov3-repo',
    '--split', 'val', '--device', 'cpu'
)
python -m omnishield benchmark @edgeArgs --frames 50 --warmup 5 --out runs/edge-deployment-v5/cpu-benchmark.json
python -m omnishield export @edgeArgs --out runs/edge-deployment-v5/omnishield-edge-v5.onnx
```

Benchmark replay đo từ đọc ảnh đã ghi tới khi đầu ra có mặt trên máy chủ. Báo cáo gồm p50, p95, p99 và thời gian từng khung. Kiểm TensorRT FP16, đo luồng camera thật và nghiệm thu theo phần 9 của [đặc tả triển khai](docs/omnishield_x_implementation_spec.md).

## Replay 360° trên trình duyệt

Dùng cùng `$edgeArgs` để chạy demo cục bộ. Demo cần PyTorch, SciPy và scikit-image.

```powershell
python -m omnishield demo @edgeArgs --port 8766
```

Mở [màn hình replay](http://127.0.0.1:8766/). Góc nhìn 360°, camera đã nắn méo và vạch dẫn hướng cập nhật từ ảnh ghi. Lớp phủ AR và PDC ước tính hiện khi AI trả kết quả cho đúng khung và cấu hình camera. Nếu CPU suy luận chậm hơn nhịp replay, tạm dừng để xem lớp phủ của khung đang chọn.

Trong Profile 1, bật hoặc tắt **Độ sâu LiDAR** để so sánh trên cùng một khung. Chế độ này dùng bản quét PHENIKAA và cần nhóm phụ thuộc `phenikaa`. Ảnh ghi cung cấp màu, còn LiDAR hỗ trợ chỉnh mặt chiếu gần vật thể và ưu tiên camera có góc nhìn ít bị che.
