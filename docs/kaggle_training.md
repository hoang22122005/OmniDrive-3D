# Huấn luyện OmniShield-X trên Kaggle

## 1. Chuẩn bị các input

| Input | Tệp tối thiểu |
|---|---|
| Source OmniShield | `omnishield-src.zip` và file Notebook `kaggle/OmniShield.ipynb` |
| PHENIKAA | Clip có `camera/CAM_F_*/<timestamp>.jpg`, `lidar/<timestamp>.laz`, `label/<timestamp>.txt` nếu có |
| Hiệu chuẩn | `Camera_Intrinsics.json`, `Sensor_Extrinsics.json` cùng bản dataset. Điền `configs/rig/phenikaa.example.json` với clip train/val/test, transform LiDAR→ego đã kiểm, mặt đất và footprint thực |
| Backbone | Bản source Meta DINOv3 có `hubconf.py` cùng package `dinov3/`, checkpoint Meta cho `dinov3_vits16` có quyền truy cập. Dùng cùng source/revision cho mọi run so sánh |
| Prepared | `manifest.json`, `audit.json`, `lut/<rig>.npz`, `labels/<drive>/<frame>.npz`, do job prepare hoặc adapter ngoài tạo |

Giữ các bản dữ liệu và trọng số có điều khoản hạn chế ở phạm vi truy cập phù hợp với quyền đã được cấp. Gói source chạy với các input đã đính kèm. Job `prepare` dùng NumPy, Pillow và `laspy[lazrs]`. Job train sử dụng PyTorch có CUDA của runtime Kaggle. Nếu thiếu `laspy/lazrs`, cài trong Notebook hoặc đính kèm wheel phù hợp với runtime. [PHENIKAA devkit](https://github.com/phenikaa-dataset/phenikaa-devkit) · [Trọng số DINOv3](https://github.com/facebookresearch/dinov3#pretrained-models) · [Kaggle Python runtime](https://github.com/Kaggle/docker-python)

## 2. Thứ tự chạy

1. **Smoke trên Kaggle.** Import notebook, gắn source ZIP, bật GPU và chạy `JOB='smoke'`. Profile này dùng CNN nhỏ và dữ liệu dựng để thử backward, lưu `last.pt` và chấm validation.
2. **Chuẩn bị một phần dữ liệu thật.** Chọn ít nhất một clip train và một clip val từ các chuyến độc lập. Điền `RIG_CONFIG`, cập nhật `RAW` và `CALIBRATION`, rồi chạy `JOB='prepare'`. Lưu output `/kaggle/working/phenikaa-prepared` thành input Kaggle có version. Kiểm tra ảnh chiếu LiDAR theo devkit tác giả trước khi tăng quy mô.
3. **Train Edge trên GPU.** Bật GPU và đính kèm source, dữ liệu gốc, prepared data và DINOv3. Chọn `JOB='edge-42'`. Giữ `RAW` và `PREPARED` cùng release. Mặc định dùng một GPU, AMP FP16, batch 1 và tích gradient 4.
4. **Resume qua phiên Kaggle.** Lưu output gồm cả `last.pt` và `best.pt`. Ở phiên tiếp theo, đặt `CHECKPOINT` tới `last.pt` đã lưu và giữ JOB, seed, dữ liệu, source DINOv3 và precision. Chương trình từ chối resume khi chữ ký run khác. `MAX_MINUTES` để lại thời gian lưu output. Mỗi checkpoint được ghi ở biên optimizer step.
5. **Fine-tune cấu hình final supervised.** Bổ sung target `health`, `velocity`, `risk`, `center` và `offset` cho train/val. Dùng `JOB='final-supervised'` và `INITIAL_CHECKPOINT` trỏ tới `best.pt` của Edge. Cách này nạp weights rồi tạo optimizer và run mới.
6. **Đánh giá.** Chọn `JOB='evaluation'`, đặt `CHECKPOINT` tới `best.pt` và chốt split trước. CLI `predict` ghi NPZ voxel từ RGB, camera mask và LUT. CLI `export` dùng CPU, cần `onnx` và `onnxruntime`, so đầu ra với PyTorch trên sample và ca mất toàn bộ camera. Việc build TensorRT và đo Orin thực hiện với phần cứng đích.

Bảy công việc đã cấu hình gồm smoke, chuẩn bị dữ liệu, ba seed độc lập, fine-tune và đánh giá. Chạy seed thứ hai và thứ ba khi pipeline seed 42 ổn định. Notebook sử dụng một worker ghi output cho mỗi run. Checkpoint dùng để nối tiếp một run theo thứ tự giữa các phiên. [Hướng dẫn Notebook Kaggle](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)

## 3. Hợp đồng dữ liệu và nhãn

Manifest v1 chứa `source`, `revision`, `rigs` và `frames`. Mỗi frame có `drive_id`, `frame_id`, `timestamp_ns`, `split`, `rig_id` và mảng đường dẫn ảnh tương đối theo thứ tự trước, sau, trái, phải. Slot mất ảnh dùng `null`. Audit chặn drive nằm ở nhiều split, calibration không khớp ảnh, thiếu file và target sai shape. Chữ ký resume gồm cấu hình, manifest, hash LUT/nhãn, source huấn luyện và DINOv3, nguồn trọng số và chế độ AMP. Giữ nguyên phiên bản ảnh gốc đã gắn.

| Target trong NPZ | Shape | Quy ước |
|---|---|---|
| `occupancy` | [Z,Y,X] | 0 free, 1 occupied, −1 ignore |
| `semantics` | [Z,Y,X] | ID lớp theo model config, −1 ignore. CE chỉ trên occupied có lớp |
| `depth` | [C,H/4,W/4] | Bin hit theo range, bin cuối là escape, −1 ignore |
| `health` | [C] | Điểm sức khỏe camera [0,1] |
| `velocity`, `offset` | [2,Z,Y,X] | Vận tốc m/s và offset tới tâm theo mét trong hệ ego |
| `risk`, `center` | [Z,Y,X] | Nhãn trong [0,1] |

Với trường float, NaN đánh dấu vị trí thiếu nhãn. Mỗi label cần giữ provenance. Adapter PHENIKAA khớp timestamp theo tên file, dùng điểm LiDAR để tạo depth và các hit occupied, đồng thời gán lớp theo hộp có nhãn phù hợp. Chỉ bật `confirmed_single_lidar_origin` khi gốc tia đúng cho điểm trong scan. Nếu scan là point cloud gộp và thiếu gốc tia từng điểm, dùng nhãn free có nguồn gốc rõ ràng hoặc bộ chuẩn hóa riêng. Trainer kiểm đủ support free và occupied trong phần camera phủ.

Có thể gắn ParkOcc hoặc OccTrack360 qua manifest `source=normalized` và nhãn đúng lưới, FoV, lớp. Phép chiếu hỗ trợ pinhole, KB4 và MEI. Adapter raw cho hai bộ này cần chuyển đổi theo giao thức riêng trước khi chạy. Profile Edge bốn view dành cho rig bốn camera theo spec.

## 4. Phần đã có trong cấu trúc và phần nghiên cứu tiếp theo

Đường train hiện có backbone Meta DINOv3 đóng băng, depth theo bin, gộp evidence không âm, lớp voxel có điều kiện, head theo nhãn bổ sung, checkpoint/resume và đánh giá mIoU/coverage. `final_supervised` áp dụng tiêu chí đủ nhãn cho các head, sử dụng dữ liệu một khung hình. Chất lượng phụ thuộc dữ liệu và cần đo trên split độc lập.

Các bước nghiên cứu sau baseline gồm temporal với ego-motion, ghép instance xuyên chuỗi, Oracle/KD 7B, G-ray/FiLM, Fisher-EDL/Yager, QAT và benchmark chuẩn theo từng bộ. Hợp đồng cho các bước đó nằm trong đặc tả. Sa bàn F01–F12 vẫn dùng nguồn mô phỏng riêng trong khi chưa có checkpoint và tích hợp replay RGB thật.
