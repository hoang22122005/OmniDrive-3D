# Danh mục dữ liệu huấn luyện & đối chiếu OmniShield-X

Tài liệu này tổng hợp các bộ dữ liệu phục vụ ba mục tiêu: **(1) dựng MVP**, **(2) huấn luyện bản Pro giàu ngữ nghĩa trước khi đưa ra thực tế**, và **(3) so sánh đối chiếu (benchmark)** với các phương pháp nghiên cứu liên quan.

---

## 1. Nhóm dữ liệu Fisheye 360° (Khớp trực tiếp phần cứng 4 camera)

| Bộ dữ liệu | Cảm biến & Nhãn chính | Vai trò trong dự án | Trạng thái tích hợp |
|---|---|---|---|
| **PHENIKAA** | 4 fisheye (`KB4`), 5 LiDAR, 10 camera, box 3D | **Chủ lực MVP & Pro**: bối cảnh đường Việt Nam (xe máy đông, hẻm, vỉa hè), dùng cả cho Stage 0 Oracle (5L+10C) | Đã có adapter raw ([phenikaa.py](../omnishield/data/phenikaa.py)) |
| **ParkOcc** | 4 fisheye quanh xe, semantic occupancy bãi đỗ | Bổ sung không gian hẹp, hầm/bãi đỗ cho **F08, F09, F10**; đối chiếu *AdaptiveOcc v2* | Cần adapter → `source="normalized"` |
| **OccTrack360** | 4 fisheye, chuỗi dài, 4D panoptic occupancy & tracking ID | Bổ sung tracking xuyên chuỗi, vận tốc `m/s`, mask che khuất cho **F01–F04**; đối chiếu *FoSOcc* | Cần adapter → `source="normalized"` |
| **WoodScape / SynWoodScape** | 4 fisheye quanh xe, LiDAR, box 3D, semantic, **lens soiling** | Train **F11 (`health` head)** trên nhãn bẩn/giọt nước thật & dữ liệu CARLA; đối chiếu *FisheyeBEVDet* | Cần adapter → `source="normalized"` |
| **Fisheye3DOD** | 4 fisheye quanh xe, box 3D | Tăng khả năng xử lý méo cầu ở rìa ảnh fisheye | Cần adapter → `source="normalized"` |

---

## 2. Nhóm dữ liệu Multi-Camera & Occupancy quy mô lớn (Pre-train & Đối chiếu)

Dùng để pre-train tăng độ phủ ngữ nghĩa 3D (tránh thiếu mẫu ở các lớp ít gặp) và so sánh với các mô hình SOTA:

| Bộ dữ liệu | Cảm biến & Nhãn chính | Vai trò trong dự án | Bài báo đối chiếu tiêu biểu |
|---|---|---|---|
| **Occ3D-nuScenes / OpenOccupancy / SurroundOcc** | 6 camera pinhole, LiDAR, dense 3D semantic occupancy + `mask_camera` | **Pre-train ngữ nghĩa 3D** quy mô lớn (1.000 cảnh); đối chiếu benchmark quốc tế | *FlashOcc, FastOcc, ALOcc, SparseOcc, OPUS-V2, RadOcc* |
| **IDD-3D** | 6 camera quanh xe, LiDAR (~12.000 frame), box 3D & tracking | **Bổ sung mật độ xe máy/scooter châu Á** len lỏi không theo làn cho lớp `rider`, `pedestrian` | Benchmark giao thông đô thị châu Á |
| **Occ3D-Waymo** | Multi-cam, LiDAR, dense occupancy | Đối chiếu mở rộng quy mô lớn | *Occ3D, SparseOcc* |
| **KITTI-360** | 2 fisheye 180° + stereo trước + LiDAR 360°, Semantic Scene Completion | Bổ sung vật cản tĩnh, tường, bó vỉa (`curb`), cây | *S2GO* |
| **SemanticKITTI** | Stereo/LiDAR, Semantic Scene Completion | Đối chiếu chuẩn SSC cổ điển | *ProtoOcc, OPUS* |

---

## 3. Nhóm dữ liệu kiểm thử độ bền, mô phỏng & thích nghi xe thật

| Nguồn dữ liệu | Đặc điểm | Vai trò trong dự án |
|---|---|---|
| **PHENIKAA-Corrupted** | 8 lỗi × 3 mức (24 điều kiện: bùn, mưa, lóa, thiếu sáng, nhòe, frame cũ, blackout, lệch gá) | Train **Stage 2 KD** (Teacher sạch → Student nhiễu) và đo `mRR` / `mCE`, `health` head (F11) |
| **CARLA Simulator** *(chạy local)* | Dữ liệu tổng hợp có GT tuyệt đối (depth, semantic, ID, vận tốc, vùng khuất) | Sinh thêm các ca hiểm khó thu ngoài đời (F03 tạt đầu, F05 Hidden-Rider, F06/F07 vật dưới gầm) và kiểm thử mô phỏng |
| **Video 360° thực tế trên xe (USB)** | 4 camera fisheye thực tế của xe đích (kèm calibration thật & pose/CAN nếu có) | **Fine-tuning Stage 3A** (trộn 50% video USB + 50% dữ liệu có nhãn) và kiểm thử trước khi triển khai |

---

## 4. Lộ trình sử dụng đề xuất

1. **MVP (`edge-42`):** Dùng riêng **PHENIKAA** (`JOB='prepare'` → `JOB='edge-42'`) với 5 lớp (`free`, `static`, `curb`, `rider`, `pedestrian`).
2. **Pre-train mở rộng ngữ nghĩa (tùy tài nguyên GPU):** Pre-train trên **Occ3D-nuScenes** + **IDD-3D** để tăng độ giàu ngữ nghĩa và nhận diện xe máy.
3. **Train bản Pro trên hệ 4 Fisheye:** Gộp **PHENIKAA + ParkOcc + OccTrack360 + WoodScape** (quy đổi qua `source="normalized"` về cùng lưới voxel và taxonomy 5–6 lớp), kết hợp **PHENIKAA-Corrupted** (Stage 2 KD) và ca hiểm từ **CARLA** nếu cần.
4. **Fine-tuning chốt hạ cho xe thật (Stage 3A → 3B QAT):** Trộn **50% video 360° thực tế từ xe (USB) + 50% dữ liệu fisheye có nhãn** với đúng calibration của xe đích, sau đó chạy QAT và benchmark TensorRT trên phần cứng đích.
5. **So sánh đối chiếu:** Chạy cùng evaluator, cùng split theo `drive` độc lập và cùng ROI trên `Occ3D-nuScenes`, `ParkOcc`, `OccTrack360` và `PHENIKAA-Corrupted`.

---

## 5. Quy tắc chuẩn hóa khi thêm bộ dữ liệu mới

- Mọi bộ dữ liệu ngoài PHENIKAA phải được chuyển đổi về định dạng `manifest.json` (`version: 1`, `source: "normalized"`) và file nhãn `.npz` theo đúng hợp đồng tensor trong [kaggle_training.md](kaggle_training.md#3-hợp-đồng-dữ-liệu-và-nhãn).
- Chia `train` / `val` / `test` theo **`drive_id` độc lập** trước khi tạo cửa sổ thời gian hoặc sinh nhiễu (`audit` sẽ chặn nếu một drive xuất hiện ở nhiều split).
- Phép chiếu hỗ trợ sẵn `pinhole`, `kb4` và `mei` trong [calibration.py](../omnishield/geometry/calibration.py); mỗi camera bắt buộc có `K`, hệ số méo và `T_cam_from_ego` đã kiểm chứng overlay.
