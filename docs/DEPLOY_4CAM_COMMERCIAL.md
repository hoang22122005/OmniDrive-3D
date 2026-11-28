# Hướng Dẫn Triển Khai Thương Mại Camera 360° 4-Cam (OmniShield-X)

Tài liệu này mô tả trọn bộ quy trình đóng gói phần cứng + đo nội tham số KB4 + in bạt hiệu chuẩn gara + chạy luồng live 4 camera và xuất bảng tra LUT cố định 60 FPS (**không dùng LiDAR, không cần AI backbone khi vận hành**).

---

## 1. Cấu Hình Phần Cứng Đề Xuất (BOM ~6–8 triệu VND)

| Thành phần | Thông số đề xuất | Ghi chú |
|---|---|---|
| **4× Camera góc siêu rộng** | AHD 1080p/720p hoặc GMSL2, cảm biến **Sony IMX307 / IMX225**, ống kính fisheye **190°–195°** (IP68) | Mua cùng 1 lô lens → chỉ cần đo nội tham số KB4 **1 lần duy nhất** |
| **Đầu thu 4 kênh** | **Lựa chọn A (dễ test nhất):** Card thu **4-in-1 Quad-Split AHD → USB 3.0 UVC** (xuất 1 luồng 2×2 ghép 4 cam)<br>**Lựa chọn B (độ nét cao):** 4× USB UVC / MIPI-CSI (4 kênh độc lập `/dev/video0..3`) | OmniShield hỗ trợ trực tiếp cả 2 chế độ (`--quad` hoặc `--sources`) |
| **Board xử lý 60 FPS** | **Jetson Orin Nano** / **Orange Pi 5 (RK3588)** / **Android Box (Allwinner T5 / Snapdragon 665)** / PC Windows+Linux có OpenGL ES 3.0 hoặc WebGL 2 | Runtime chỉ chạy phép chiếu + bảng tra LUT cố định (dùng < 5% GPU) |
| **Bạt hiệu chuẩn gara** | In bạt Hiflex trắng mờ chống chói **4 tấm 1.20 m × 1.20 m** (đã sinh sẵn 1:1 ở Bước 3) | Dùng chung cho mọi dòng xe sau khi nhập chiều dài/rộng xe |

---

## 2. Bước 1 — Đo Nội Tham Số Lens KB4 (Làm 1 Lần Cho Đúng Mã Lens Mua Về)

In bảng checkerboard **9×6 ô trong** (kích thước mỗi ô `40 mm` trên giấy A3/A2 dán phẳng lên tấm formex cứng), quay 1 video 20–30 giây đưa bảng phủ đều khắp tâm và 4 rìa góc 190° của camera, rồi chạy:

```bash
# Từ thư mục ảnh checkerboard hoặc file video quay sẵn
python -m omnishield calib-intrinsics \
  --video data/calib/lens190_checkerboard.mp4 \
  --cols 9 --rows 6 --square-mm 40 \
  --slot all \
  --out runs/Camera_Intrinsics.json \
  --verify-img runs/kb4_straightness_check.jpg

# Hoặc chụp trực tiếp từ camera USB đang cắm (nhấn SPACE chụp 20-35 góc, nhấn Q để giải)
python -m omnishield calib-intrinsics \
  --camera 0 --cols 9 --rows 6 --square-mm 40 \
  --slot all \
  --out runs/Camera_Intrinsics.json \
  --verify-img runs/kb4_straightness_check.jpg
```

- Tiêu chuẩn đạt: `RMSE < 0.50 px` và ảnh kiểm tra `runs/kb4_straightness_check.jpg` hiển thị cạnh bảng thẳng tắp không cong vồng.
- File `runs/Camera_Intrinsics.json` xuất ra đúng định dạng KB4 (`fx, fy, cx, cy, k1..k4`) của OmniShield-X.

---

## 3. Bước 2 — In 4 Tấm Bạt Hiệu Chuẩn 1:1 & Sơ Đồ Trải Gara

Sinh 4 file bạt vector **SVG** (đem ra tiệm in quảng cáo in đúng tỉ lệ 1:1 kích thước `1.20 m × 1.20 m`, không co giãn), ảnh PNG kiểm tra, sơ đồ trải dây quanh xe và file tọa độ 3D chuẩn cho kích thước xe thực tế:

```bash
# Ví dụ cho xe dài 4.75 m, rộng 1.934 m (thay thông số xe của bạn)
python -m omnishield make-mats \
  --length 4.75 --width 1.934 --mat-size 1.20 \
  --out runs/garage_mats
```

Sản phẩm sinh ra trong `runs/garage_mats/`:
- `mat_FL.svg`, `mat_FR.svg`, `mat_RL.svg`, `mat_RR.svg` (4 tấm bạt in Hiflex 1:1, có thước kiểm chứng `100 mm` và 4 vòng tròn đánh dấu `P1–P4`).
- `garage_layout_blueprint.svg` (sơ đồ đo thước dây đặt 4 tấm bạt chính xác quanh thân xe).
- `mat_world_coords.json` (tọa độ `(X, Y, Z)` chuẩn mét của các góc bạt cho bộ giải PnP).

---

## 4. Bước 3 — Chạy Live 4 Camera & Căn Ngoại Tham Số 6-DoF Bằng Thảm (PnP)

Khởi chạy server thu luồng live trực tiếp từ phần cứng (thay thế hoàn toàn thư mục ảnh tĩnh):

```bash
# Cách A: Đầu thu 4-in-1 Quad-Split AHD USB (1 thiết bị 2x2)
python -m omnishield live \
  --quad 0 --quad-order 0,1,2,3 \
  --profile runs/calib_lidar_profile.json \
  --lut-out runs/lut_export \
  --port 8765 --open

# Cách B: 4 camera USB / V4L2 / RTSP / video độc lập (thứ tự: Trước, Sau, Trái, Phải)
python -m omnishield live \
  --sources 0,1,2,3 \
  --profile runs/calib_lidar_profile.json \
  --lut-out runs/lut_export \
  --port 8765 --open
```

Trên giao diện trình duyệt (`http://127.0.0.1:8765/`):
1. Bấm **⚙ Chỉnh 4 Cam** → chọn lần lượt từng tab **Trước / Sau / Trái / Phải**.
2. Kéo 4 ghim **P1–P4** (có kính lúp zoom 3.2× tự hiện khi kéo) vào đúng 4 góc bạt caro thật trên nền gara.
3. Bấm **✓ Giải PnP 6-DoF** cho từng camera → kiểm tra sai số chiếu lại `< 1.0 px` và vạch bạt nối liền phẳng trên lòng chảo 3D rộng (`pad_x = 9 m, pad_y = 7 m`).
4. Bấm **💾 Lưu Profile** (lưu `calib_profile.json`) và bấm **📦 Xuất LUT nhúng** để sinh bảng tra cố định ngay trên giao diện.

---

## 5. Bước 4 — Xuất Bảng Tra LUT Cố Định Cho Thiết Bị Nhúng (60 FPS, Zero Calib Runtime)

Có thể chạy từ nút **📦 Xuất LUT nhúng** trên UI hoặc bằng CLI:

```bash
python -m omnishield export-lut \
  --profile runs/calib_lidar_profile.json \
  --out runs/lut_export \
  --width-2d 384 --height-2d 512
```

Các file đóng gói sẵn trong `runs/lut_export/`:
- `avm_2d_lut.bin` / `avm_2d_lut.npz`: Bảng tra 2D Top-Down (`384×512`, magic `AVM2`, `uint16 u/v` + `uint8 weight` 4 camera).
- `avm_3d_bowl_lut.bin` / `avm_3d_bowl_lut.npz`: Lưới lòng chảo 3D rộng (`pad_x = 9 m, pad_y = 7 m`, magic `AVM3`, đỉnh OpenGL XYZ + `uint16 u/v` + `uint8 weight` 4 camera + index buffer).
- `avm_lut_spec.h`: File header C/C++ định nghĩa struct nhị phân chuẩn cho OpenGL ES 3.0 / Vulkan trên Android NDK, RK3588, Allwinner T5, Jetson Orin Nano.
- `lut_manifest.json`: Tóm tắt kích thước & thông số profile đã bake.
