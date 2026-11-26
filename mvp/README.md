# OmniShield-X · MVP 3D

Mở `index.html` bằng Chrome hoặc Edge. Các tệp chạy trực tiếp từ thư mục này và dùng bốn camera fisheye RGB-D ảo. Có thể chọn một trong tám kịch bản hoặc bấm F01–F12 để chuyển tới kịch bản có tính năng đó.

## Xem và kiểm tra

1. Mở **Xe máy tạt đầu**, chọn F01, F03 hoặc F04 và kéo thanh thời gian quanh 2 s. F04 hiển thị gia tốc ứng viên theo mô phỏng.
2. Chọn F06 để xem điểm vật thấp đã lưu khi xe đi qua. Mở **Lớp bản đồ và giới hạn** rồi bật **Khung tham chiếu** để đối chiếu vật thể của cảnh.
3. Chọn **Mất tín hiệu** ở một camera. Nhãn **Đã tách** cho biết camera đã bị loại. Vùng có camera khác quan sát vẫn giữ bằng chứng của camera đó. **Bùn che** và **Lóa đèn** dùng cửa sổ 1,5 s. Có thể phát replay để cửa sổ tiến theo thời gian.
4. Dùng **Lưu hộp đen** rồi **Mở hộp đen** để xem lại 5 giây voxel cùng track và TTC mô phỏng. Khi replay đi qua sự kiện TTC dưới 1,2 s, nút lưu dùng sự kiện đầu tiên làm mốc. Tệp JSON sử dụng run-length encoding và ghi kích thước thực khi xuất.

## Phạm vi

- Sa bàn dùng lưới XY 20 cm và 4 dải cao. Voxel được tính từ khoảng cách theo tia và nhãn pixel của camera ảo. Các hộp tham chiếu chỉ tham gia dựng dữ liệu và đo sai lệch. Cấu trúc voxel mang trạng thái rời rạc.
- Nguồn RGB-D mô phỏng cung cấp độ sâu, class và tình trạng cảm biến. Demo dùng bộ phân nhóm đơn giản với tối đa một đối tượng động mỗi loại trong thiết kế cảnh. Đây là mức thử cho thuật toán hình học và hậu xử lý. Đường suy luận RGB-only bằng OmniShield-Edge, Dirichlet, đánh giá camera thật và TensorRT thuộc spec chính.
- Với ba dòng bị cắt trong yêu cầu, dùng ngưỡng demo cho F02 là vận tốc dưới 0,3 m/s với vùng cửa ngang 0,94–2,15 m, F03 là vận tốc ngang hướng vào trục trên 0,35 m/s, và F09 là lùi với tốc độ quay trên 0,21 rad/s.
- F05 dùng tỷ lệ ô có bóng che làm điểm rủi ro hình học. F06 lưu điểm thấp theo khóa 5 cm trong tọa độ thế giới. F08 hiện thảm dự báo và kiểm các hướng thay đổi trên vùng có bằng chứng. F10 đánh dấu ô ứng viên cùng độ phủ. F11 lọc tình trạng cảm biến mô phỏng. Điểm F12 là điểm minh họa theo tỷ lệ frame vượt ngưỡng TTC. Tệp chưa chứa dữ liệu GPS thực.
- Số đo **Hợp nhất CPU** chỉ gồm dựng voxel trong trình duyệt. **Sai lệch bề mặt TB** là khoảng cách từ tâm các voxel có hit tới bề mặt hộp gần nhất trong cảnh mô phỏng. Bảng đo này phục vụ kiểm hình học và thời gian tại chỗ.

Chạy kiểm thử từ gốc workspace bằng Node.js đã cài đặt:

```powershell
node --test mvp/replay.test.js
```

[Đặc tả kỹ thuật](../docs/omnishield_x_implementation_spec.md)
