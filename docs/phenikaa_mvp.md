# MVP được huấn luyện bằng dữ liệu thật

MVP của OmniShield-X gồm checkpoint DINOv3 đã train trên camera thật, báo cáo trên một clip giữ riêng và replay suy luận chỉ từ bốn ảnh RGB cùng calibration. Dự đoán lưu occupancy theo bốn dải cao. Bản đồ BEV của replay là hình chiếu từ voxel. Các mô-đun temporal, Oracle và 12 tính năng nguyên bản có dữ liệu và tiêu chí nghiệm thu riêng.

## Dữ liệu đã xác minh

| Chia tập theo clip | Clip PHENIKAA Mini | Khung có đủ dữ liệu | Thành phần theo khung |
|---|---|---:|---|
| Train | COMPLEX CITY | 200 | 4 ảnh fisheye, LiDAR LAZ, hộp 3D, tệp ego pose |
| Validation | COMPLEX URBAN | 200 | 4 ảnh fisheye, LiDAR LAZ, hộp 3D, tệp ego pose |

Hai clip cách nhau hơn một tháng theo timestamp trong tên file. Bản MVP có một clip train và một clip validation. Chất lượng ngoài hai clip cần thêm các chuyến độc lập và một tập test khóa riêng. Catalog khóa frame ID và byte count cho **2.802 tệp, khoảng 663 MB raw**, gồm cả calibration. Chương trình có thể chạy lại để lấy các tệp chưa tải. [PHENIKAA Mini](https://drive.google.com/drive/folders/1G490BK62MlmYhMMlapE_Smm6JgO3WmnJ) · [Giấy phép và devkit](https://github.com/phenikaa-dataset/phenikaa-devkit#license)

## Nhãn và phạm vi của MVP

- Đầu vào gồm bốn ảnh fisheye 1920 × 1080 khớp theo timestamp gốc. Model dùng 320 × 576, lưới đều 20 cm và vùng khoảng 16 × 12 m. Gốc lưới thử nghiệm ở giữa vị trí hai camera trước/sau suy ra từ extrinsics. Mask thân xe bảo thủ và mặt đất tham chiếu z=0 dùng để tạo nhãn cho hai clip.
- Nhãn **occupied** đến từ các điểm LiDAR trên mặt đất tham chiếu. Nhãn sâu lấy điểm chiếu gần nhất theo từng ô ảnh. Ngữ nghĩa vehicle, rider và pedestrian lấy từ hộp 3D phù hợp và chỉ gán trên voxel có hit.
- Nhãn **free** là nhãn xấp xỉ. LAZ mẫu có `point_source_id=0` cho toàn bộ điểm nên pipeline dùng nhiều ô độ sâu LiDAR chiếu lên ảnh, chừa biên khoảng cách trước các điểm trả về và giữ unknown khi thiếu hỗ trợ. `audit.json` ghi nguồn free của từng rig và số nhãn theo lớp. Báo cáo IoU đo theo các nhãn xấp xỉ cùng quy ước.
- Model nhận ảnh và bảng tra rig. LiDAR, pose và hộp 3D đi qua đường chuẩn bị nhãn. MVP hiện huấn luyện một khung hình và replay theo trình tự thời gian của các frame.

## Chạy trên Kaggle

1. Đính kèm **omnishield-src.zip** và import **OmniShield.ipynb**. Nếu dùng gói dữ liệu địa phương đã chuẩn bị, đính kèm riêng **phenikaa-mvp.zip** và trỏ `DATA_ZIP` tới file. Nếu chuẩn bị từ Kaggle, bật Internet, cài `laspy[lazrs]`, chạy `JOB='prepare-mvp'` và lưu output `/kaggle/working/phenikaa-mvp` làm input có phiên bản. `MVP_FRAMES_PER_CLIP=4` chỉ dùng để kiểm đường dữ liệu.
2. Kiểm `prepared/audit.json` và `data-review.png`. Chọn `JOB='mvp-real'`, bật GPU và gắn source Meta DINOv3 cùng checkpoint `dinov3_vits16` có quyền truy cập. Cấu hình ban đầu dùng 6 epoch, batch 1 và tích gradient 4. Lưu `best.pt` và `last.pt`. [Checkpoint DINOv3](https://github.com/facebookresearch/dinov3#pretrained-models)
3. Chọn `JOB='replay-mvp'`, đặt `CHECKPOINT` tới `best.pt` và giữ `SPLIT='val'`. Notebook ghi `metrics.json`, dự đoán voxel NPZ và `rgb-replay.gif`. Replay hiển thị vùng chưa quan sát khi camera chưa phủ hoặc độ bất định vượt ngưỡng hiển thị.
4. Nghiệm thu sau khi có checkpoint bằng occupancy IoU, support và IoU từng lớp có nhãn, ảnh dự đoán trên clip giữ riêng, và xác nhận vùng mất camera hiển thị hợp lý. Khi bổ sung cảnh thật, giữ đánh giá về độ chính xác và tốc độ trong phạm vi dữ liệu đã đo.
