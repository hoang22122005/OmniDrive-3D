# OmniShield-X · Đặc tả triển khai pipeline

**Phiên bản 3.0 · 25/11/2026 · Phạm vi nghiệm thu đầu tiên: nghiên cứu offline và replay.**

**Bàn giao đầu tiên** là một clip PHENIKAA đi hết đường dữ liệu: bốn ảnh cùng mốc thời gian → hiệu chuẩn → nhãn có nguồn gốc → dự đoán voxel của Edge → so sánh PyTorch với TensorRT → báo cáo sai số và độ trễ. Dùng một rig, một lưới đều và một checkpoint cho toàn bộ phép so sánh. Một lỗi ở mốc nào phải tái hiện được bằng chính clip và cấu hình của mốc đó.

Tài liệu này quy định cách triển khai từ bản thiết kế nghiên cứu `omnishield_master_spec_v2.md`. Các con số lựa chọn trong spec là mặc định cho thí nghiệm hoặc ngưỡng nghiệm thu đề xuất. Khi phát hành bản này, workspace có đặc tả, còn kết quả trên dữ liệu thật và phần cứng đích đang chờ đo.

## 1. Phạm vi và cấu hình đầu tiên

| Quyết định | Edge một khung hình | Mở rộng sau khi có kết quả đối chứng |
|---|---|---|
| Cảm biến khi suy luận | 4 fisheye theo thứ tự trước, sau, trái, phải. Ảnh RGB 384 × 640 sau tiền xử lý. Có cờ hợp lệ từng camera | Thêm pose và timestamp vào bộ nhớ thời gian. Oracle dùng 5 LiDAR và 10 camera ở phía huấn luyện |
| Backbone | DINOv3-S/16, RoPE gốc, đủ 960 patch, 1 CLS và 4 register. Backbone đóng băng ở lần chạy đầu | LoRA gộp tĩnh, pruning, G-ray, FiLM, NAF. Pro dùng S+/16 |
| Không gian | Cartesian đều 160 × 160, 20 cm/ô, phủ 32 × 32 m, bốn dải cao | Lưới 5 cm cục bộ sát thân xe. Đánh giá phần giao theo mét |
| Đầu ra | Bằng chứng trống/đặc, độ bất định, semantic theo **từng voxel**, trạng thái quan sát | ID theo voxel xuyên chuỗi, vận tốc m/s và nguy cơ xuất hiện từ vùng khuất |
| Phần cứng | FP32 làm tham chiếu. FP16 batch 1 trên Orin NX 16 GB là đích replay. RTX 4060 Laptop được đo riêng | INT8 có Q/DQ, distillation 7B và Pro đều có thí nghiệm riêng |
| Mục tiêu thời gian | Đề xuất **p95 ≤ 25 ms** từ lúc nhận đủ bộ frame đến khi có sự kiện đầu ra. Ghi cả p99 và tỷ lệ trễ hạn | Đo riêng độ trễ từ phơi sáng đến cảnh báo khi đồng hồ cảm biến đã đồng bộ. Mục tiêu Edge 20,8 ms compute và 22,3 ms sau nhận frame của master là giả thuyết cần đo |

Cấu hình được khóa ở đầu mỗi run qua `resolved_config.yaml`. File chứa phiên bản schema, revision dữ liệu, seed, checkpoint và hash, rig, split, lưới, bin độ sâu, taxonomy, các module đang bật, runtime và cấu hình cảnh báo. Tên run không thay thế cho các trường này.

## 2. Các gói công việc và sản phẩm bàn giao

| Mốc | Phụ thuộc | Phạm vi triển khai | Điều kiện chuyển mốc |
|---|---|---|---|
| M0 · Dữ liệu | Có clip và calibration của cùng rig | Bộ đọc, manifest, chia tập, đồng bộ, phép chiếu, bảng tra và overlay | Một clip có thể replay theo thứ tự với đủ trạng thái cảm biến. Đạt kiểm thử hình học ở §4 và có overlay đã duyệt |
| M1 · Nhãn | M0 | Nhãn chiều sâu, occupancy, semantic, nguồn nhãn và mask. Tạo tập kiểm tay độc lập | Đạt các cảnh dựng ở §5. Có thống kê độ phủ nhãn theo lớp, dải cao, cự ly và camera |
| M2 · Edge FP32 | M1 | Backbone một khung hình, lifting, fusion và head voxel. Huấn luyện và đánh giá lần đầu | Overfit được mẫu nhỏ, xuất được cả lỗi dự đoán, vượt đối chứng đơn giản trên metric chính của validation (§10) |
| M3 · Edge FP16 | Có hợp đồng tensor M2 | Thử export toán tử sớm ngay từ M0 bằng tensor dựng. Sau M2 đóng gói checkpoint thật và replay trên Orin | Đạt parity, có log từng frame và có kết luận đạt/trượt ngân sách thời gian (§9) |
| M4 · Temporal và cảnh báo replay | M2, M3 | Ego-warp, bộ nhớ có hạn dùng, tracking, risk, hậu xử lý và lỗi cảm biến | Qua các ca reset, mất cam, giao cắt và đổi ID. Báo cáo bỏ sót, cảnh báo nhầm và lead time theo sự kiện |
| M5 · Mở rộng nghiên cứu | Baseline đo được cùng evaluator | Lưới gần, Pro, F-EDL, Oracle, thích nghi video và QAT | Từng biến thể có run đối chứng, chênh lệch chất lượng, peak VRAM và số đo trên thiết bị tương ứng |

## 3. Dữ liệu, đồng bộ và hợp đồng mẫu

PHENIKAA công bố 5 LiDAR, 10 camera và nhãn hộp 3D. Adapter phải kiểm kê bản phát hành thực tế để xác định tên kênh, camera model, độ phân giải, timestamp, pose, ID và mức phủ của annotation. Lưu giấy phép nguồn cùng revision trong manifest. PHENIKAA công bố CC BY-NC-SA 4.0. [PHENIKAA devkit và dữ liệu](https://github.com/phenikaa-dataset/phenikaa-devkit)

### 3.1. Các bảng đầu vào

| Bảng / khóa | Trường bắt buộc và cách dùng |
|---|---|
| `rig.json` theo `rig_id, calib_hash` | Mỗi cảm biến có tên gốc, slot chuẩn, model chiếu, hệ số méo, kích thước ảnh gốc, miền tia hợp lệ, transform 4 × 4, đơn vị. Lưu footprint thân xe theo mét và offset từ gốc xe tới tâm footprint |
| `frames.jsonl` theo `sequence_id, frame_id` | `drive_id, split, rig_id, calib_hash, t_ref_ns`. Mỗi camera có `path, capture_ns, clock_id, status`. Mỗi LiDAR có `path, start_ns, end_ns, sensor_id`. Lưu các tham chiếu pose, box/track và annotation nếu có |
| `poses` theo `sequence_id, time_ns` | `T_world_from_ego`, chất lượng và nguồn pose. Chuẩn hóa chiều transform, đơn vị và clock lúc ingest. Nội suy tịnh tiến tuyến tính và quaternion SLERP trong khoảng có dữ liệu |
| `split.json` và `label_manifest` | Danh sách drive mỗi tập, lý do loại, phiên bản luật tạo nhãn, hash đầu vào, nguồn và mức tin cậy từng loại nhãn. Phiên bản mới tạo tập đầu ra mới để có thể tái hiện run cũ |

Dùng số nguyên 64 bit cho timestamp ns. Trường thiếu có giá trị null và mã lý do, ví dụ `missing_file`, `clock_unmapped`, `stale`, `decode_error`. Adapter xuất đúng bốn slot kể cả khi một camera hỏng. Với nguồn tham chiếu đa cảm biến thiếu kênh, ghi số kênh thực dùng vào metadata của nhãn.

### 3.2. Quy tắc ghép và chia dữ liệu

1. Quy đổi mọi timestamp sang đồng hồ chung theo rig. Lịch replay lấy các mốc `t_ref` trong manifest. Với mỗi slot, chọn frame mới nhất có `capture_ns ≤ t_ref_ns`. Khi hai frame trùng thời điểm, phá hòa theo ID nguồn. Bộ đọc online và offline dùng chung luật. Ngắt chờ sau ngân sách đồng bộ đã cấu hình theo chu kỳ rig, phát đủ bốn slot với mask cho cam thiếu, ghi thời gian chờ và các frame tới muộn.
2. Tính `age_i = (t_ref_ns - capture_ns) / 1e9`. Mặc định thử nghiệm `max_camera_age = 50 ms`, `max_pair_skew = 25 ms`. Cam quá tuổi mất hiệu lực. Cặp vượt skew tắt stereo cho cặp đó. Ghi p50/p95 độ lệch và tỷ lệ loại theo slot để chọn lại ngưỡng trên tập train/validation của rig.
3. Lưu `T_ego(t_ref)_from_sensor(t_capture)` cho nhãn có pose đủ tốt. Ở Edge một khung hình, dùng extrinsics tĩnh tại frame đã đồng bộ và đo sai số do skew. Nếu sai số vượt dung sai nhãn ở §4, siết ngưỡng ghép hoặc bù pose theo camera và dựng bảng lấy mẫu tương ứng. LiDAR tích lũy cần bù pose từng sweep, ưu tiên deskew theo thời gian từng điểm nếu có.
4. Chia theo drive **trước** khi tạo cửa sổ, tích lũy sweep, học chuẩn hóa và sinh corruption. Dùng split của tác giả khi đánh giá benchmark. Với PHENIKAA nội bộ chưa có split phù hợp, dùng hash ổn định của drive và seed để chia thử 70/15/15, kiểm lại số sự kiện Rider và cảnh đêm ở mỗi tập. Mọi bản sạch, nhiễu và pseudo-label của một drive đi cùng tập.
5. Chuỗi train có `T=1` cho baseline, `T=4` cho temporal. Cửa sổ dừng ở ranh drive, đổi calibration hoặc mất timestamp. Cờ `seq_start` kích hoạt reset. Kiểm tra tự động rằng tải mẫu tại thời điểm t chỉ truy cập ảnh student có timestamp ≤ t.

### 3.3. Tiền xử lý ảnh

Decode thành RGB, letterbox giữ tỷ lệ đến 384 × 640, ghi riêng tỷ lệ và padding từng trục. Pixel được biểu diễn tại tâm pixel. Với resize hệ số s, hoành độ tâm mới là `s*(u+0.5)-0.5+pad_x`, tương tự cho v. Chuẩn hóa ảnh RGB [0,1] bằng mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225). Vùng pad và ngoài vòng ảnh được gắn `pixel_valid=0`. Cam lỗi đưa tensor ảnh chuẩn hóa bằng 0 và `sensor_valid=0`. Kiểm parity CPU/GPU bằng ảnh có điểm màu đánh dấu đủ bốn góc.

## 4. Hình học, lưới và bảng tra

Quy ước `T_A_from_B` nhân vector cột ở B để thu tọa độ ở A. Chọn ego tại tâm trục sau của rig (X tiến, Y trái, Z lên). Nếu dataset đặt gốc chỗ khác thì lưu phép đổi gốc tường minh, áp dụng cho cả nhãn, pose và footprint. Hệ camera quang học có x phải, y xuống, z theo trục quang. Dùng transform khai báo của từng rig để chuyển giữa hai hệ. Độ sâu tia là khoảng cách Euclid từ tâm camera, đơn vị mét. Phép chiếu theo model đã hiệu chuẩn quyết định tia hợp lệ, kể cả tia có thành phần z âm ở rìa ảnh.

| Thành phần | Định nghĩa dùng chung cho nhãn, model và evaluator |
|---|---|
| Lưới nền | X và Y thuộc [−16,16) m. Chỉ số `ix=floor((x+16)/0.2)`, `iy=floor((y+16)/0.2)`. Tâm X là `−16+(ix+0.5)*0.2`. Tensor lưu theo `[z,iy,ix]`. Điểm đúng biên trên thuộc ngoài lưới |
| Chiều cao | Các cạnh Z bằng [−0.2, 0.4, 1.0, 1.7, 2.5] m. Mỗi khoảng đóng trái, mở phải. Lifting baseline lấy mẫu tại tâm từng dải. Nhãn occupancy ghi occupied nếu có bề mặt hợp lệ trong dải, lưu độ phủ để đánh giá sai số của phép lấy mẫu tâm |
| Độ sâu | D=32 khoảng. 20 khoảng đều 0,2 m từ 0,2 đến 4,2 m. 12 khoảng tiếp theo có cạnh `4.2*(32/4.2)^(j/12)` với j từ 1 đến 12. Một kết quả bổ sung biểu diễn tia đi qua toàn tầm. Điều kiện `0.2 ≤ r < 32` và miền chiếu quyết định mask voxel/camera |
| Lưới 5 cm thử nghiệm | Lưới Cartesian phụ lấy footprint đã đo, nới 2,2 m mỗi phía rồi làm tròn biên ra ngoài theo bội số 5 cm. Với footprint ví dụ `X=[−2.1,+2.1] m`, `Y=[−0.9,+0.9] m`, thu **172 ô X × 124 ô Y**. Mask thân xe riêng. Lưới nền vẫn phủ vùng xa. Trong phần giao, đầu ra lưới phụ sở hữu voxel. Hai lưới đọc cùng feature ảnh và nâng chiều độc lập để giữ chi tiết 5 cm. Rasterize cả hai trên miền vật lý chung cho evaluator |

Lựa chọn hai lưới xử lý trực tiếp giới hạn của thiết kế cũ. Với trục X có 160 ô, lõi 18 ô và hai vành 44 ô cho mỗi bên, phần ngoài mỗi bên chỉ còn 27 ô để phủ 11,7 m, trung bình 43,3 cm/ô. Huấn luyện và đo chi phí lưới phụ trước khi quyết định dùng cho Edge.

**LUT theo rig.** Chiếu tâm voxel về tọa độ ảnh đã letterbox. Với feature stride 4, đặt `u4=(u+0.5)/4−0.5` rồi chuẩn hóa `gx=2*(u4+0.5)/160−1`, tương tự cho v với chiều cao 96. Dùng `align_corners=False`. Lưu `uv[C,Z,Y,X,2]`, `depth_bin[C,Z,Y,X]`, `range_m`, `fov_valid`, mask phần thân xe và bốn chỉ số/trọng số song tuyến UV. Tâm voxel chọn đúng một khoảng độ sâu theo bảng cạnh. Dùng chung luật này trong tham chiếu PyTorch và TensorRT. Tất cả toán hạng ở ngoài miền bị mask trước fusion. Hash LUT gồm calibration, phép resize, footprint, lưới, bin và phiên bản mã chiếu.

**Nghiệm thu M0.** Kiểm tra tâm, biên ảnh, rìa FoV trên 90°, pad, bốn góc BEV, từng mặt biên Z và một điểm ngoài lưới. Với fixture giải tích FP32, sai số project→unproject→project tối đa 0,1 px trên miền hợp lệ đã chọn, tâm ô→mét→ô trả đúng chỉ số cho mọi ô. Kiểm độc lập transform bằng dịch 1 m và quay 90°. Trên clip thật, chồng điểm LiDAR lên ảnh từng camera ở đầu, giữa, cuối clip. Ghi sai số theo cự ly tại mốc tĩnh nhận diện được và sai số quy đổi trên mặt đất. Lưới 5 cm chỉ được tuyên bố có độ chính xác định vị tương ứng khi đã đo được sai số tổng hợp trên vùng đó.

## 5. Sinh nhãn và kiểm soát chất lượng

| Đại lượng | Cách tạo target | Mask và ca phải kiểm |
|---|---|---|
| Chiều sâu camera | Chiếu điểm LiDAR đã bù thời gian, dùng z-buffer giữ **range gần nhất nhìn thấy từ camera**. Kiểm occlusion do khoảng cách giữa tâm LiDAR và tâm camera. Gán bin theo §4. Chỉ xác nhận escape khi đường tia camera tới biên có bằng chứng khoảng trống hoặc hit đã kiểm chứng nằm ngoài tầm. Chỉ gán nhãn tia đi hết tầm khi có chứng cứ quan sát đủ tầm | Pixel không có LiDAR mặc định `ignore`. Pseudo-depth từ UniK3D hoặc nguồn bổ sung mang cờ nguồn, confidence và sai số sau hiệu chỉnh theo mét |
| Occupancy vật lý | Raycast từ **gốc từng LiDAR** đến điểm trả về. Ghi free ở phần đã đi qua, occupied ở điểm dừng. Cắt tia tại biên ROI. Các ô sau vật cản giữ unknown cho tia này, có thể được cảm biến khác quan sát | Mã 0 free, 1 occupied, 255 unknown/ignore. Ưu tiên hit đã kiểm chứng khi một voxel chứa cả mặt và không khí. Lưu hit count, free count và nguồn. Bản đầu cho free khi có đoạn tia hợp lệ đi qua và không có hit trong voxel, đánh dấu đây là free quan sát thưa để đánh giá riêng trên tập kiểm tay. Bỏ các điểm tự phản xạ của thân xe |
| Visibility của student | Chiếu voxel vào từng fisheye. So range với bề mặt đầu tiên đã kiểm chứng trong cùng hướng. Lưu `visible, occluded, outside, unlabelled` riêng cho từng cam | Một voxel có thể occupied, khuất ở cam trái và nhìn thấy ở cam sau. Lưu mask nhãn độc lập với mask runtime |
| Semantic và ID | Free lấy từ nhãn hình học. Voxel có bề mặt gán một trong Static, Curb/Low obstacle, Rider, Pedestrian dựa trên annotation, track và xác minh ảnh. Chỉ dùng ID khi liên kết đối tượng đủ tin cậy | ID 0 cho stuff/free, ID dương duy nhất trong sequence cho từng thing, 255 là ignore của **semantic**. Voxel occupied thiếu lớp vẫn dùng cho loss hình học. Vật thể ngoài taxonomy được đánh dấu ignore cho semantic |
| Vận tốc | Từ track cùng ID trong hệ world, sai phân vị trí theo thời gian thực, xoay vector về ego hiện tại. Unit m/s. Ưu tiên pose/track có chất lượng khai báo | Cần ít nhất 2 timestamp hợp lệ. Mục tiêu là vận tốc vật thể so với mặt đất, biểu diễn trong ego. Vật đứng yên có target 0 kể cả khi ego di chuyển |
| Lens health | Với lỗi tổng hợp có mask pixel, target health bằng 1 trừ mức che/hỏng đã khai báo. Mất hẳn cảm biến được xử lý bằng cờ cứng. Clip thật dùng vùng được người kiểm duyệt | BCE chỉ trên pixel có target. Vùng chưa gán nhãn health phải được bỏ khỏi loss |
| Hidden-Rider | Tại t, đánh dấu footprint của Rider/Pedestrian khuất với tất cả camera student hợp lệ nhưng được nguồn độc lập xác nhận. Mẫu dương khi track đi vào hành lang footprint xe quét trong **1 s kế tiếp**, theo quỹ đạo ego tham chiếu | Mẫu âm cần đủ quan sát oracle xuyên hết horizon. Cố định thời điểm cảnh báo đầu ra tại t, dựng hành lang GT từ pose tương lai chỉ trong bộ sinh nhãn. Mất track, hết clip, thiếu pose hoặc vùng không thể xác nhận mang mask ignore. Lưu ID, thời điểm bắt đầu khuất và lần đầu vào hành lang |

Đối với nhãn tĩnh, có thể tích lũy sweep từ t−2 đến t+2 trong cùng split sau khi đưa về ego tại t và loại vật chuyển động. Với vật động, chỉ dùng sweep hiện tại hoặc bù theo track đã xác minh. Tương lai dùng để tạo **target** và chấm sự kiện. Tạo một dataset view riêng cho student chỉ có ảnh, metadata và state nhân quả.

**Tập kiểm tay.** Lấy clip sạch, clip đông Rider, một vùng góc giao camera, cảnh đêm và trường hợp che khuất. Khóa danh sách theo sequence trước khi tối ưu mô hình. Ghi số voxel/track/sự kiện có nhãn thật, pseudo và ignore. Đánh giá chính trên phần được kiểm độc lập, công bố điểm trên pseudo-label kèm tên nguồn. Fixture tối thiểu gồm tia không hit đã quan sát hết tầm, hai bề mặt cùng pixel, mặt phẳng tĩnh, một Rider sau xe đỗ và hai vật khác lớp cùng tọa độ XY nhưng khác Z.

## 6. Mạng Edge và hợp đồng tensor

B là batch, C=4 camera, Z=4 dải cao, Y=X=160. Engine chạy với B=1. Tiền xử lý host giữ timestamp ở int64 và đưa tuổi ảnh tính bằng giây sang bộ kiểm hợp lệ. CNN/ViT có thể gộp B×C thành batch nội bộ nhưng phải hoàn nguyên đúng thứ tự cam trước lifting.

| Tên | Shape và dtype tại giao diện | Ý nghĩa |
|---|---|---|
| `images, sensor_valid` | [B,C,3,384,640] float32 tham chiếu hoặc float16, [B,C] bool | Ảnh chuẩn hóa và trạng thái sau kiểm timestamp/decode |
| `pixel_valid` | [B,C,1,96,160] bool | Mask stride-4 gồm vòng ảnh, padding và vùng bị loại đã biết. Khi nội suy dùng trọng số chỉ trên pixel hợp lệ |
| `ray_logits, gain, health` | [B,C,33,96,160], hai tensor [B,C,1,96,160] | 32 hit + 1 escape, gain không âm, health trong [0,1]. Baseline cố định gain=8, health=1 ở pixel hợp lệ. Các cổng cứng vẫn có hiệu lực |
| `F_sem, F_bev` | [B,C,72,96,160], [B,128,Y,X] | Đặc trưng ảnh và BEV. Bốn dải cao giữ thứ tự từ thấp lên cao |
| `mass, p_occ, observed` | [B,3,Z,Y,X], [B,1,Z,Y,X], [B,1,Z,Y,X] | Mass FP32 tham chiếu theo thứ tự free, occupied, ignorance. `observed` là bool theo bằng chứng của frame hiện tại |
| `sem_cond, sem_prob` | [B,4,Z,Y,X], [B,5,Z,Y,X] | Phân bố lớp có điều kiện đã occupied và xác suất chung theo thứ tự Free, Static, Curb/Low obstacle, Rider, Pedestrian |
| `center, offset, velocity` | [B,1,Y,X], [B,2,Z,Y,X], [B,2,Y,X] | Mở ở M4. Center heatmap, offset mỗi voxel tới tâm đối tượng (mét XY), vận tốc tại tâm theo trục ego |
| `instance_id, risk` | [B,Z,Y,X] int32 sau tracker, [B,1,Y,X] float | ID theo voxel và xác suất sự kiện có horizon 1 s. Output public kèm `valid`, tuổi trạng thái và timestamp |

### 6.1. Từ ảnh đến BEV

1. Chạy `dinov3_vits16` với RoPE gốc, lấy patch của block 12 sau chuẩn hóa. Bỏ prefix CLS/register, reshape thành [B×C,384,24,40]. Chiếu 1×1 xuống 96 kênh và upsample bilinear lên 96 × 160. Nhánh chi tiết gồm hai Conv3×3 stride 2 lần lượt 16 rồi 32 kênh, BN và SiLU. Concat thành 128 kênh. Hai head 1×1 sinh `F_sem` và 33 ray logits. Giữ config và checksum của checkpoint. [DINOv3](https://github.com/facebookresearch/dinov3)
2. Softmax FP32 trên 33 kết quả. Với khoảng k, đặt `hit=p[k]`, `free=p[escape]+sum(p[j] cho j>k)`, `behind=sum(p[j] cho j<k)`. Tổng ba thành phần bằng 1. Tia đi hết tầm cho free=1 tại mọi bin. Bin ngoài miền là invalid. Làm fixture hit ở bin thứ hai để bắt lỗi đảo chiều tổng.
3. Lifting lấy `F_sem` từ UV bằng bilinear 2D. Với mỗi voxel và từng góc UV, lấy free/hit/behind đúng bin range của voxel rồi nội suy theo UV. Nhân các trọng số hình học với mask và chuẩn hóa lại trên các góc có support. Khi tổng support bằng 0, trả zero và invalid. Có thể cài bằng bốn gather với chỉ số tĩnh, dùng chung tham chiếu cho export.
4. Đặt `reliability = sensor_valid × fov_valid × pixel_support × health`, trong đó `pixel_support` là tổng trọng số song tuyến hợp lệ trước chuẩn hóa. Gộp đặc trưng từng voxel bằng trung bình có trọng số `reliability × hit`. Mẫu số bằng 0 cho vector zero. Chiếu 72→32 theo dải, concat theo Z thành `F_bev`. Head 3×3 từ BEV sinh 4×Z logits, reshape và softmax trên bốn lớp conditional.

### 6.2. Bằng chứng, xung đột và trạng thái thiếu quan sát

Mỗi camera tạo hai evidence không âm `eF=gain*free`, `eO=gain*hit`, `S=2+eF+eO`. Discount theo reliability r bằng `f=r*eF/S`, `o=r*eO/S`, `u=1−f−o`. Pixel đứng sau mặt chắn hoặc camera mất tín hiệu có thể đóng góp mass toàn phần vào ignorance. Với gain học được, dùng `clamp(softplus(z),0,25)`. 

Baseline một khung hình cộng evidence đã nhân r theo camera và tính Dirichlet nhị phân với prior (1,1). Trả `mass=(ΣreF, ΣreO, 2)/(2+ΣreF+ΣreO)`. Ở thí nghiệm Yager dùng mass (f,o,u) từng cam. Tích dưới đây chạy qua **tất cả** cam tại voxel và chuyển conflict một lần ở cuối, kể cả nơi có hơn hai cam.

```text
U0 = product(u_i)
F  = product(f_i + u_i) - U0
O  = product(o_i + u_i) - U0
K  = 1 - F - O - U0
mass = (F, O, U0 + K)
p_occ = O + 0.5 * mass.ignorance
```

Áp dụng cùng phép chuyển `p_occ=mass.occupied+0.5*mass.ignorance` cho baseline. Tính `sem_prob = concat(1−p_occ, p_occ×sem_cond)`. `observed` bằng true khi có ít nhất một cam với `r*(eF+eO)>eps`, mặc định eps=1e−6. Xuất cả mass và cờ. Khi mọi cam invalid, fusion phải cho (0,0,1), `p_occ=0.5`, `observed=false`; hậu xử lý xuất trạng thái chưa quan sát. Khi xung đột tuyệt đối, Yager cũng cho (0,0,1) và ghi K=1, lúc đó các phép đo có tồn tại. Tính tích/tổng ở FP32. Kiểm tổng mass sai lệch tối đa 1e−5, hoán vị cam giữ kết quả trong dung sai và cam invalid là phần tử trung hòa.

### 6.3. Các mô-đun nghiên cứu có công tắc cấu hình

| Mô-đun | Thay đổi cần cài | Phép đối chứng |
|---|---|---|
| LoRA và pruning | Rank theo nhóm ba block là 0, 8, 16, 32. Với rank dương đặt alpha=2r cho qkv/proj. Merge trọng số trước export. Giữ tất cả patch giao ROI theo rig và biên pitch. Lấy số slot cố định bằng số patch cần giữ lớn nhất giữa bốn cam, pad và mask key trước softmax, mask query trước scatter | So logits trước/sau merge ở FP32. Mục tiêu 672 patch/cam được thử riêng sau khi kiểm coverage sát xe và trên dốc |
| G-ray và FiLM | Đối chứng đầu tiên tái lập RoPE gốc theo đúng thứ tự H,W, tâm patch, tọa độ chuẩn hóa và periods của checkpoint. Biến thể góc dùng `atan2(dy,dz), atan2(dx,dz)`, scale mỗi trục thành `2*theta/FoV` trước bước tính pha, kiểm gián đoạn ở rìa. Với FiLM, tính metric bằng Jacobian 3×2 của **tia đơn vị** theo pixel và phân rã JᵀJ. Dùng bốn sin/cos góc, log diện tích, log tỷ số trục và cặp cos/sin góc kép, triệt hướng khi hai trị riêng bằng nhau | So RoPE gốc với thay góc trên cùng checkpoint. FiLM khởi tạo gamma=beta=0. Kiểm giá trị hữu hạn ở rìa, ngoài vòng ảnh và tâm. [Mã RoPE của Meta](https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/rope_position_encoding.py) |
| NAF, ALOcc và stereo | Lấy tap 3/6/9/12, chiếu về cùng chiều. NAF×4 chuẩn hóa mask bằng số pixel hợp lệ trong cửa sổ với kernel đếm cố định. Gate dùng RMSNorm và tanh chặn biên. ALOcc thử conv tách biệt trên 32 hit logits trước softmax, giữ kênh escape. Stereo dùng 16 kênh chia 4 nhóm, cosine ở voxel thấy được đồng thời và đạt ngưỡng skew. Gain stereo dùng softplus để evidence không âm | Bật từng nhánh, ghi sai số depth ở bốn góc, recall vùng gần, bộ nhớ và latency thực tế |
| Lưới gần và Pro | Lưới gần dùng cùng bộ phép chiếu/đầu ra ở §4, tính loss độc lập theo mét và mask sở hữu lưới. Pro thay backbone bằng S+/16 giữ cùng số cam và ROI | Đo riêng chi phí lưới phụ và thay backbone. So near-field trên cùng tập tia, cùng voxel GT và cùng hành lang quanh thân xe |

## 7. Bộ nhớ thời gian, tracking và cảnh báo trong replay

**Giao diện state.** Engine temporal nhận thêm `H_prev[B,128,Y,X]`, `vel_prev[B,2,Y,X]`, `dyn_prev[B,1,Y,X]`, `age_prev[B,1,Z,Y,X]`, `T_prev_from_cur[B,3,3]` (SE(2)), `dt_s[B]` và `reset[B]`. Cả bốn tensor state được trả lại cho lần gọi kế tiếp, age tính bằng giây ở FP32. Runtime giữ state trên GPU theo từng stream/rig. Tracker giữ bộ đếm ID, vị trí, vận tốc và lần cuối quan sát ở host. Khi reset, xóa tracker, cho H/vel/dyn bằng 0 và age bằng giá trị quá hạn.

1. **Bù ego.** Với từng tâm ô hiện tại, biến đổi sang hệ ego trước, lấy bilinear H và vận tốc tại ô nguồn. Xoay vector vận tốc về ego hiện tại bằng phần quay nghịch đảo, chỉ cộng tịnh tiến cho **điểm**. Warp age theo nearest, ngoài ROI đặt quá hạn. Khi `dt≤0`, đổi sequence/rig, pose hỏng hoặc `dt>max_gap_s`, reset bộ nhớ và dùng phép đo hiện tại. Mặc định replay `max_gap_s=0.5` trên PHENIKAA, thiết lập lại theo cadence từng dataset.
2. **Đối chứng temporal.** Dùng H đã warp ở ô còn hạn và trộn với `F_bev` bằng độ tin cậy `1−u` của từng dải cao. Với nhóm 32 kênh của dải z, đặt `H_new=(1−g)*H_warp+g*F_current`, `g=observed*(1−u)`. Lần quan sát đầu tiên của ô hết hạn khởi tạo từ `F_current` nếu đạt ngưỡng update. Age chỉ về 0 khi có quan sát tin cậy (`g≥0.2` mặc định), ngược lại tăng theo dt. Hết `memory_ttl_s=0.5` mà chưa có cập nhật thì vô hiệu hóa nhóm kênh. Xuất riêng hỗ trợ hiện tại và hỗ trợ từ bộ nhớ.
3. **Biến thể GDFusion.** Sau warp, chọn tối đa 512 ô thing động có score cao ở `dyn_prev` đã warp, score khởi điểm bằng tổng xác suất Rider và Pedestrian. Các slot pad mang trọng số 0. Dịch tâm bằng `v*dt` theo mét, splat bilinear và chuẩn hóa bằng tổng trọng số nhận được. Chuyển cả mask và tuổi của đặc trưng đi cùng vị trí, gỡ đóng góp động ở nguồn để tránh bóng kép. Với nhiều nguồn, age lấy giá trị lớn nhất trong các nguồn thực sự đóng góp. Chiếu tương quan cửa sổ 5×5 tại đích thành residual 128 kênh qua linear 25→128. Corrector dùng `H_pred + g*sigmoid(Conv1x1([H_pred,F_current]))*tanh(Conv1x1(F_current−H_pred)+residual)`. So với temporal đối chứng trên các ca chéo camera và drift.
4. **Đọc state.** Head semantic đọc H mới. Ở biến thể temporal, thêm head occupancy nhị phân có softplus để dự đoán cặp evidence từ H cho các voxel còn hạn. `p_occ` phục vụ semantic/tracking lấy từ head temporal với prior (1,1). Lưu mass quan sát hiện tại của §6 độc lập để kiểm tra xung đột và chặn cảnh báo. Khi voxel hết hạn và không còn quan sát, xuất prior và `valid=false`. Loss occupancy của head dùng mask vật lý ở §5. Ca xe đứng yên, ego quay 90°, vật đi qua biên ROI, tất cả cam blackout lâu hơn TTL và hồi phục từng cam phải chạy trên cùng fixture.
5. **Tracking.** NMS 3×3 trên center heatmap, ngưỡng ban đầu 0,3. Với voxel thing hợp lệ, cộng offset mét vào XY và gán tâm cùng lớp gần nhất trong bán kính khởi điểm 0,75 m. Tâm không ghép được có trạng thái unassigned. Nối tâm giữa hai frame bằng Hungarian trên khoảng cách mét sau ego-warp và dự báo vận tốc, chặn ghép khác lớp, hòa theo ID nhỏ hơn. Ngưỡng ghép khởi điểm `1 m + 3 m/s × dt`. Track hết TTL đóng lại. Ghi ID, khoảng cách ghép, số lần đổi ID, cả các lần mất track. Với hai đối tượng cùng lớp có cùng tâm XY và chồng theo Z, head center 2D có thể gộp ID. Ghi riêng số ca này. Nếu benchmark yêu cầu phân biệt, dùng center/offset 3D hoặc instance queries rồi chạy lại evaluator. Với benchmark có taxonomy khác, dùng nhóm thing và ngưỡng của adapter đã khóa.
6. **Risk và sự kiện.** Head sigmoid từ H ra risk mỗi ô BEV. Runtime nhận footprint xe và quỹ đạo dự kiến 1 s từ CAN/VO hoặc mô hình vận tốc/yaw rate hiện hành. Tính hợp của footprint tại các mốc cách 0,1 s. Với replay có thể nới biên theo `margin_m = base_m + gain_m*u`, các hệ số chọn trên validation. Dùng max risk trong vùng hành lang còn hiệu lực cho sự kiện Rider. Xuất `track_id, t_ref, level, score, health_status, support_age_s` và polygon. Ba mức thử nghiệm là theo dõi, chú ý, cảnh báo. Chọn hai ngưỡng score trên validation, vào mức cao ngay khi vượt ngưỡng, giữ tối đa theo TTL và nhả sau 3 frame liên tiếp dưới ngưỡng. Với mất toàn bộ quan sát hoặc mất quỹ đạo hợp lệ, phát trạng thái suy giảm riêng và ghi vùng mất quan sát. Đối với nguy cơ từ vùng khuất, head được dùng ngữ cảnh các ô lân cận nhìn thấy hoặc bộ nhớ còn hạn. Ghi nguồn hỗ trợ của dự báo riêng với visibility của chính voxel khuất. Chấm cả những sự kiện bị bỏ sót do thiếu hỗ trợ. Mọi ngưỡng thời gian cần được kiểm lại khi thay FPS.

## 8. Huấn luyện, distillation và tài nguyên

### 8.1. Hàm mất mát

Mỗi loss tính tổng trên phần tử có nhãn rồi chia cho tổng trọng số hợp lệ của **cùng task**. Nếu mẫu số bằng 0, trả loss bằng 0 với gradient hữu hạn, đồng thời ghi bộ đếm support. Khi train phân tán, cộng cả tử và mẫu trên toàn batch logic. BCE trên xác suất phải clamp theo epsilon hoặc dùng dạng logits ổn định tương đương.

| Task | Loss và trọng số khởi đầu | Tập phần tử tham gia |
|---|---|---|
| Occupancy và semantic | BCE(`p_occ`, occupied) ×1, CE lớp conditional ×1, Lovász trên `sem_prob` ×0,5 | Occupancy biết từ sensor. Conditional CE chỉ ở voxel occupied có lớp, Lovász chỉ nơi có nhãn ngữ nghĩa |
| Depth và health | CE 33 kết quả ×0,8. BCE health ×0,2 khi bật head. Với pseudo-depth mềm dùng KL với phân bố đã căn theo đúng cạnh bin | Pixel có hit/escape xác minh. Health dùng mask độc lập. Ghi tỷ lệ supervised pixel thực |
| Panoptic và velocity | Focal loss center ×1, smooth-L1 offset theo mét ×0,5, smooth-L1 vận tốc ×0,3 | Center target Gaussian với sigma khởi điểm 0,2 m, rasterize theo lưới. Offset chỉ trên voxel thing có ID. Vận tốc chỉ tại tâm có track đủ quan sát |
| Hidden-Rider | BCEWithLogits ×0,5 trên mask risk. Cân bằng positive theo tỷ số âm/dương tính trên train, chặn trọng số ở 10 | Nhãn sự kiện theo §5, ghi số mẫu dương mỗi epoch. Theo dõi precision–recall ở cấp sự kiện |
| I-EDL và KD | Giai đoạn F-EDL thử I-EDL trên alpha=(1+eF,1+eO) của nguồn có nhãn visibility hợp lệ, loss Fisher/logdet tính FP32, KL tăng tuyến tính 0→0,08. Giữ loss occupancy gộp. Dùng hàm tham chiếu đã pin và đối chiếu giá trị/gradient bằng fixture | Chỉ bật sau baseline CE. Stage KD dùng trọng số khởi điểm 1,5 BEV, 0,8 ray, 0,5 Gram. [I-EDL](https://proceedings.mlr.press/v202/deng23b.html) · [Mã tác giả](https://github.com/danruod/IEDL) |

### 8.2. Lịch thí nghiệm đề xuất

| Run | Khởi tạo và lịch | Điều kiện sử dụng |
|---|---|---|
| Smoke và Edge một frame | Overfit 8 mẫu có cả free/occupied. AdamW, LR head 1e−4, weight decay 0,01. Sau smoke chạy 12 epoch, warmup 1 epoch, cosine. B hiệu dụng 4, microbatch 1 rồi tích gradient nếu cần | Bắt đầu với backbone đóng băng, loss depth + occupancy + semantic. Chọn checkpoint trên validation. Thử LoRA với LR 2e−4 ở run riêng |
| Temporal | Khởi tạo từ Edge. T=4, B hiệu dụng 4 chuỗi, gradient clip 1, LR 5e−5, tối đa 12 epoch. Truyền state theo thứ tự trong clip, detach giữa các cửa sổ | Bật loss tracking, velocity và risk khi nhãn tương ứng có support. Fit riêng ngưỡng tracking và cảnh báo trên validation |
| Stage 0 Oracle | DINOv3-7B đóng băng. Dùng ảnh sạch theo từng camera, chiếu đặc trưng ảnh lên cùng lưới, voxelize 5 LiDAR cùng thời điểm, SparseConv và concat BEV, neck 1×1→128 kênh. Train neck/head occupancy, semantic, ray khoảng 5 epoch, LR 2e−4 | Chỉ dùng trong huấn luyện và đánh giá teacher. Lưu metric teacher theo vùng có nhãn thật để xác nhận chất lượng target |
| Stage 2 KD | Student nhận ảnh corrupted, teacher nhận cùng mẫu sạch. Trong 12 epoch LR 5e−5, dùng L1 giữa BEV sau project 1×1 và chuẩn hóa kênh, KL phân bố ray teacher→student, MSE Gram của patch chuẩn hóa L2 trên các cặp cùng tọa độ | BEV cùng frame/grid, ray target cùng bin, Gram chỉ trên các patch giao của cùng camera. Với augmentation hình học phải warp cả target và mask. Nhánh 6 camera pinhole bổ sung đặc trưng BEV cho teacher. Biến thể RadOcc đầy đủ thêm loss affinity bằng MSE hai ma trận cosine của feature trên cùng cặp vùng semantic hợp lệ ở ảnh fisheye, lấy trung bình với ray KL và ghi hai loss riêng |
| Stage 3A và 3B | Adapt video 3 epoch LR 2e−5, replay 50% video đích và 50% dữ liệu train có nhãn. QAT 3 epoch LR 1e−5, teacher EMA, hệ số EMA khởi điểm 0,999 | Video chỉ có ảnh sử dụng Gram và consistency khi có pose/calibration phù hợp. Loss RadOcc-ray cần target render hợp lệ. Sau adapt phải đo lại tập đêm và tập gần xe trước QAT |

**Đo tài nguyên.** B=4, T=4 và 10 camera tương đương 160 lượt ảnh teacher cho một bước logic. Thử microbatch 1 camera, 1 frame, rồi tăng tuần tự. Dùng `no_grad` cho teacher đóng băng ở Stage 2. Ở Stage 0 chỉ backbone teacher dùng `no_grad` để neck vẫn nhận gradient. Đo một chu kỳ đủ forward, backward, optimizer step sau warmup, bao gồm EMA/optimizer states. Log allocated/reserved và peak theo từng GPU, thời gian bước, ảnh/s, dtype, camera/chunk. Với 2×T4 đo cả thời gian chuyển target. Nếu vượt VRAM, giảm microbatch, tách GPU, hoặc cache target theo `sequence/frame/calib/teacher_hash` và đo cả dung lượng cache. Cho phép thử checkpoint teacher nhỏ hơn như một cấu hình đối chứng. Mục tiêu dưới 16 GB chỉ đạt khi đo đúng card, batch logic và chuỗi đã công bố.

**Augmentation.** Lỗi ảnh dùng cặp clean/corrupt có cùng timestamp, giữ seed cố định cho validation. Biến đổi hình học phải cập nhật đồng thời ảnh, ray, intrinsics/extrinsics tương ứng, LUT, footprint nhãn và vector vận tốc. Với scale 3D, đổi các đại lượng mét theo cùng phép biến đổi, kiểm riêng tác động đến kích thước vật thể. Với lỗi drift gá camera, mô phỏng ảnh quan sát theo pose lệch trong khi model nhận hiệu chuẩn danh định, ghi cả hai vào metadata. Bật từng phép sau khi overlay vẫn đúng cho nhánh hình học hoặc thể hiện đúng sai lệch đã cài cho nhánh lỗi.

## 9. Export, so khớp và benchmark

1. **Khóa môi trường.** Lưu OS, GPU, JetPack/L4T trên Orin, CUDA, TensorRT kèm minor version, PyTorch, ONNX, ONNX Runtime, opset, plugin, flags và dtype từng binding. Chọn stack từ bản JetPack thực cài trên thiết bị. Build engine riêng cho Orin và RTX. Khóa hash của ONNX, engine, checkpoint, LUT và file cấu hình cảnh báo.
2. **Đường tham chiếu.** Eval mode, freeze BN, merge LoRA khi bật. Export Edge một frame với batch 1, bốn cam cố định. Baseline dùng opset 17 với gather tĩnh cho lifting và phép lấy mẫu 2D khi cần. Với biến thể lấy mẫu volume 5D, kiểm đúng schema/opset, exporter và parser trên máy đích bằng graph tối thiểu. Có thể tái lập trilinear bằng tám gather có trọng số. Temporal kiểm riêng TopK, pad, trùng chỉ số scatter-add và reset state. [ONNX GridSample](https://onnx.ai/onnx/operators/onnx__GridSample.html)
3. **FP16 trước INT8.** So PyTorch FP32 → ONNX Runtime → TensorRT FP16 bằng cùng đầu vào đã serialize. Với INT8, biểu diễn Q/DQ trong graph, cast rõ vùng softmax, tổng evidence và tích mass về FP32. Trên TensorRT 10.x explicit Q/DQ, dùng chế độ strong typing thích hợp với bản đã pin. QAT thử SmoothQuant alpha=0,5 và scale riêng patch/prefix sau khi đo histogram. Kiểm inspector để xác nhận kernel/precision thực tế và chi phí split token. [TensorRT explicit quantization](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html#explicit-quantization)
4. **Parity.** Lưu input, LUT, state, ray probability, mass và đầu ra cho clip chuẩn và từng ca lỗi ở §10. Dung sai ban đầu cho sai số tuyệt đối cực đại trên xác suất là 1e−4 với ONNX FP32 và 1e−2 với TensorRT FP16. Mask cứng, reset, shape, thứ tự class và timestamp phải khớp chính xác. So ID trên fixture có đối tượng tách rõ. Log cả argmax đổi gần biên quyết định, lỗi tích lũy theo thời gian, NaN/Inf và sai số theo từng head. INT8 có thêm ngưỡng đề xuất ΔmIoU giảm ≤0,2 **điểm phần trăm** so với FP32 và kiểm recall ở cùng FAR.
5. **Đo máy thật.** Pure compute dùng GPU Compute Time của `trtexec`. Đo replay bằng clock monotonic từ lúc nhận đủ bộ frame tới khi publish kết quả sau CPU tracking/hysteresis, gồm decode, tiền xử lý, copy, đồng bộ và hàng đợi phát sinh sau điểm bắt đầu. Nếu decoder chạy trước bộ ghép, ghi thời gian đó vào phép đo capture-to-alert. Đo thêm tuổi từng camera tại thời điểm xuất sự kiện. Chỉ dùng tên glass-to-alert khi có timestamp phơi sáng đã quy đổi về cùng clock. [NVIDIA benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/benchmarking.html)
6. **Giao thức báo cáo.** Batch 1, một stream replay, queue hữu hạn độ dài 1 cho bộ frame đang chờ, bỏ bộ cũ khi quá tải và ghi số drop. Chạy ít nhất 2.000 lần sau warmup cho compute và một lượt replay liên tục 10 phút sau ổn định nhiệt cho phép đo đích. Ghi p50/p95/p99, tỷ lệ >25 ms, throughput, drop rate, tuổi ảnh, power mode, clocks, nhiệt độ, điện năng nếu đo được và peak memory. Báo cáo riêng event bình thường, blackout và mật độ instance cao. Nếu giữ p95 bằng cách bỏ frame, báo cùng tỷ lệ bỏ và lead time sự kiện.

Ví dụ lệnh đo sau khi đã build engine phù hợp trên thiết bị:

```bash
trtexec --loadEngine=runs/edge_fp16/edge.plan --noDataTransfers --useCudaGraph --warmUp=1000 --iterations=2000 --duration=0
```

## 10. Đánh giá và tiêu chí nghiệm thu

### 10.1. Kiểm thử trước khi chạy dài

| Ca kiểm | Kết quả mong đợi | Bằng chứng lưu |
|---|---|---|
| Cùng một điểm ở bốn camera, đổi rig, lệch timestamp | Tâm chiếu nằm đúng theo §4. Đổi rig làm đổi hash LUT. Frame tương lai bị loại, frame stale có mask 0 | Overlay từng cam, histogram skew, log lý do loại |
| Tia escape, hit duy nhất, voxel sau hit, pixel pad, voxel ngoài tầm | Phân bố tia chuẩn hóa, behind chặn bằng chứng, invalid cho mass toàn ignorance. Nhãn vật lý và visibility vẫn đọc độc lập | Input và output kỳ vọng từng bin/voxel |
| Một cam lỗi, cả bốn cam lỗi, hai cam đối nghịch, ba cam chồng lấn | Bỏ cam lỗi cho cùng kết quả với cam trung hòa. Blackout toàn phần cho unknown. Yager bảo toàn khối lượng trong mọi thứ tự | Tensor mass, support, K, observed |
| Hai lớp trên cùng cột XY, hai track cắt nhau, ego quay tại chỗ | Semantic khác nhau giữa dải cao. ID giữ qua chuỗi dựng có vận tốc phân biệt. Vật tĩnh có vận tốc world bằng 0 | Volume GT/pred, track log theo timestamp |
| Khởi động, đổi drive, mất pose, blackout quá TTL rồi có ảnh lại | Xóa lịch sử đúng biên. Hết TTL không phát sự kiện từ track cũ. Khôi phục dựa vào quan sát mới | Trace reset/age và timeline sự kiện |
| Học mẫu nhỏ | Trên 8 mẫu sạch có nhãn xác minh, mọi loss/gradient hữu hạn. Mục tiêu smoke là giảm tổng loss ít nhất 50% so với đầu run với cùng tập và mask | Learning curve, gradient norm, phân bố lớp và số phần tử có nhãn |
| Export một và nhiều frame | Qua dung sai §9. Engine ra cùng trạng thái degraded ở fixture blackout. Frame sau reset khớp chạy từ đầu | Golden tensors, diff theo head, manifest runtime |
| Năng lực dự đoán M2 | Trên cùng split và mask đánh giá, so với predictor theo tần suất lớp học trên train. Mục tiêu ban đầu là macro mIoU và Rider recall cao hơn đối chứng, có báo cáo số mẫu cho từng lớp | Confusion matrix, số voxel/drive, điểm từng cự ly và ca lỗi tiêu biểu |

### 10.2. Bộ chỉ số và dữ liệu đánh giá

- **Occupancy và semantic.** Tính IoU nhị phân và IoU từng lớp trên voxel có GT. Báo mIoU gồm lớp nào, kèm kết quả riêng cho bốn lớp occupied và dải cao. Khoảng cách vùng gần dùng khoảng cách mặt đất tới footprint, chia [0,1), [1,3), [3,8), ≥8 m trong ROI. Cố định miền và mask GT trước khi chạy mọi model. Voxel có GT nhưng model từ chối dự đoán được ghi là abstain và tính vào bỏ sót lớp thật trong điểm nội bộ. Báo thêm độ phủ đầu ra.
- **RayIoU.** Dùng cùng gốc tia, hướng, thứ tự gặp bề mặt, taxonomy và evaluator cho mọi run. Báo ngưỡng 1/2/4 m. Với phép đo gần xe nội bộ, lọc ray theo hit GT trong vành 0–3 m quanh footprint và dùng 0,25/0,5/1 m, công bố cả số tia có hit, tia không hit và số dự đoán dương giả trên tia trống. Lưu evaluator/version và mô tả cách quy đổi lưới. [Triển khai RayIoU của SparseOcc](https://github.com/MCG-NJU/SparseOcc)
- **Panoptic và động học.** Tính mAVE bằng chuẩn L2 của sai số vận tốc m/s trên track có GT đã ghép. Ghi cả tỷ lệ ghép và số lần đổi ID. Để chạy OccTrack360, adapter dùng camera model gốc, số view thực của split, dải cao, kích thước voxel, nhóm thing/stuff và ID theo evaluator của tác giả. Trên rig khác, tạo lại LUT và head taxonomy. Nếu cần mở rộng miền Z/ROI hoặc thay head để xuất đủ các lớp của benchmark, train/eval cấu hình đó riêng. Dùng OccSQ, OccAQ, OccSTQ của evaluator. Gắn nhãn phép đo ROI nội bộ khi chỉ dùng phần giao. ParkOcc dùng bộ đo semantic occupancy trên phần dữ liệu truy cập được. [OccTrack360 và giao thức](https://arxiv.org/html/2603.08521v3) · [Dữ liệu OccTrack360](https://huggingface.co/datasets/YouthZestLin/OccTrack360) · [ParkOcc](https://github.com/yty-sky/ParkOcc)
- **Rủi ro và cảnh báo.** Một sự kiện GT là một track đủ điều kiện ở §5 đi vào hành lang tại thời điểm t*. Ghép một-một trong cùng sequence, dùng ánh xạ track dự đoán sang GT theo overlap footprint hoặc vùng giao đã kiểm tay và cửa sổ [t*−1 s,t*]. Dự đoán trùng sau lần ghép đầu tiên được ghi riêng, ghép ở cấp episode sau hysteresis. Tính AP khi quét ngưỡng risk trên các episode, recall tại FAR cố định (số episode báo nhầm/giờ replay có GT), số bỏ sót và phân bố `lead_time=t*−t_alert`. Ghi rõ khoảng thời gian thiếu GT, thời gian degraded và số sự kiện GT trong khoảng degraded. Khóa ngưỡng bằng validation trước khi chạy test.
- **Độ bất định.** ECE dùng 15 khoảng xác suất cố định trên voxel có GT occupancy. Báo reliability diagram, Brier score, coverage–risk và AUROC phân biệt vùng lỗi tổng hợp có mask với vùng sạch cùng điều kiện. Đối chứng gồm entropy softmax, Dirichlet cộng evidence, I-EDL và Yager. Báo cả clean ECE và ECE trên từng lỗi. Khi tính khoảng tin cậy, bootstrap theo drive để giữ tương quan giữa các frame.
- **Đưa cảnh báo lên xe.** Kết quả M4 nghiệm thu chức năng trong replay. Quyết định sử dụng cảnh báo thực cần hồ sơ riêng ghi tốc độ/điều kiện vận hành, giới hạn che khuất, ngân sách lead time, ngưỡng bỏ sót và FAR được chủ hệ thống chấp nhận, cách hiển thị suy giảm và số liệu thử nghiệm tương ứng. [ISO 21448](https://www.iso.org/standard/77490.html)

### 10.3. PHENIKAA-Corrupted và sáu nhóm ablation

Mỗi lỗi có 3 mức, seed cố định theo drive và frame, cùng tập sạch gốc. Giá trị trong bảng là cấu hình khởi điểm. Dùng mask cứng cho các lỗi mà runtime thực sự quan sát được qua trạng thái cảm biến hoặc timestamp. Với ảnh đen nhưng driver vẫn báo frame hợp lệ, giữ cờ driver rồi đo riêng khả năng nhận ra blackout của health head.

| Lỗi | Nhẹ / vừa / nặng |
|---|---|
| Bùn hoặc vật che ống kính | Che 20 / 35 / 50% vùng ảnh hợp lệ của một cam, giữ mask ổn định qua clip |
| Giọt mưa và màng nước | Che/phủ khúc xạ 5 / 15 / 30% diện tích hợp lệ, lưu displacement map và mask |
| Lóa đèn | Tạo vùng bão hòa 2 / 5 / 10% diện tích, giữ hướng và chuyển động liên tục |
| Thiếu sáng và nhiễu | Gain ánh sáng 0,5 / 0,25 / 0,125 trong miền tuyến tính, thêm shot/read noise với tham số được lưu |
| Nhòe | Motion kernel dài 3 / 7 / 13 px tại độ phân giải đầu vào, cùng phân bố hướng ở ba mức |
| Frame cũ | Lặp ảnh 1 / 2 / 3 chu kỳ trên một cam, giữ nguyên timestamp ảnh cũ |
| Blackout | Tắt một cam trong 0,1 / 0,5 / 2 s. Báo riêng ca phát hiện qua driver và qua nội dung ảnh |
| Lệch gá camera | Góc lệch 2 / 4 / 6 độ trên trục và chiều được ghi, giữ cố định trong một episode |

Với một metric S càng cao càng tốt ở thang [0,1], định nghĩa `mRR = mean(S_corrupt / S_clean)` qua 24 điều kiện. Với `E=1−S`, dùng `mCE = mean_lỗi(Σ_mức E_model / Σ_mức E_baseline)`, cố định baseline Edge đã công bố. Nếu mẫu số bằng 0, ghi N/A và số điều kiện tính được. Công bố thêm điểm tuyệt đối từng lỗi/mức, mức chênh clean và metric cảnh báo theo giờ.

| Nhóm | Thay đổi mỗi lượt | Tiêu chí đọc kết quả |
|---|---|---|
| 1 · Backbone | Full/pruned, S/S+, frozen/LoRA, ResNet-50 hoặc ConvNeXt-T đối chứng | mIoU, near Rider recall, compute và end-to-end |
| 2 · Hình học | Lưới đều 20 cm, thêm nhánh gần 5 cm, 32 bin log từ master hoặc bin theo §4, RoPE/G-ray, FiLM, NAF | Khóa ROI, tập tia và ngân sách ảnh. Báo các vùng ngoài tầm của cấu hình thí nghiệm |
| 3 · Lifting và temporal | Gộp một frame, bốn dải cao, group stereo, ego-warp, Top-512 và sparse cost volume | RayIoU vùng góc, mAVE, tracking, latency từng module |
| 4 · KD và QAT | Không KD, teacher chỉ camera, 5L+4C, 5L+10C+7B. So FP32, FP16, PTQ, QAT ở cùng checkpoint nguồn | Peak VRAM theo GPU, thời gian train, ΔmIoU, recall tại cùng FAR |
| 5 · Uncertainty | Entropy, Dirichlet, I-EDL, I-EDL với Yager | ECE, AUROC theo mask, abstain, false alarm và bỏ sót ở cảnh mất camera |
| 6 · Benchmark ngoài | FoSOcc cho tracking và AdaptiveOcc v2 cho ParkOcc. Thử FlashOcc, FastOcc, ALOcc, ProtoOcc, OPUS-V2, S2GO khi đã có adapter chạy được | Dùng đúng đầu ra/evaluator của từng phương pháp. Vẽ Pareto trong cùng phần cứng, ROI, số view và giao thức |

## 11. Cấu trúc mã, CLI và khả năng tái hiện

| Vị trí dự kiến | Trách nhiệm | Đầu ra chính |
|---|---|---|
| `configs/{rig,model,data,train,eval}/` | Khai báo rig, schema, taxonomy, ngưỡng, lịch thí nghiệm và lựa chọn precision | Cấu hình đã resolve, hash và revision |
| `data/{adapters,geometry,labels,augmentations}/` | Chuẩn hóa nguồn, đồng bộ, project/unproject, voxel hóa, sinh nhãn offline và corruption | Manifest, split, LUT, shard nhãn, overlay và báo cáo coverage |
| `models/{backbones,necks,lifting,temporal,heads,oracle}/` | Edge/Pro, state API, teacher đặc quyền và đầu ra đã mô tả | Checkpoint và schema binding cho từng profile |
| `losses/`, `engine/` | Masked loss, sampler, checkpoint/resume, QAT, export, replay, tracking và điều kiện cảnh báo | Engine, checkpoint, state lịch train và sự kiện |
| `eval/`, `tests/fixtures/` | Evaluator nội bộ, adapter benchmark, ca kiểm hình học, mass và state | Golden case, diff parity, metrics và log từng frame |

Ví dụ các trường cấu hình đã resolve cho lần chạy M2. Nạp các giá trị này một lần và truyền xuống bộ sinh nhãn, mô hình và evaluator.

```yaml
schema_version: 3
profile: edge_single
camera_slots: [front, rear, left, right]
image_hw: [384, 640]
grid:
  x_edges_m: {start: -16, stop: 16, step: 0.2}
  y_edges_m: {start: -16, stop: 16, step: 0.2}
  z_edges_m: [-0.2, 0.4, 1.0, 1.7, 2.5]
depth:
  linear_edges_m: {start: 0.2, stop: 4.2, bins: 20}
  log_edges_m: {start: 4.2, stop: 32.0, bins: 12}
  escape: true
fusion: additive_binary
backbone: dinov3_vits16
temporal: false
training: {clip_length: 1, microbatch: 1, accumulation: 4, seed: 42}
```

**CLI dưới đây là hợp đồng cho mã cần triển khai.** Chạy từ gốc dự án. Mỗi script kiểm tra input và ghi đầy đủ output khi thành công. Thư mục `runs/pilot` là ví dụ của một run riêng.

```text
python scripts/00_ingest.py --source phenikaa --root data/raw/phenikaa --rig configs/rig/phenikaa.yaml --out runs/pilot/data
python scripts/01_build_static_calib.py --manifest runs/pilot/data/frames.jsonl --grid configs/model/grid_20cm.yaml --out runs/pilot/calib
python scripts/02_generate_pseudo_gt.py --data runs/pilot/data --calib runs/pilot/calib --out runs/pilot/labels
python scripts/03_train_pipeline.py --config configs/train/edge_single.yaml --data runs/pilot/data --labels runs/pilot/labels --calib runs/pilot/calib --out runs/pilot/train
python scripts/04_export_and_bench.py --checkpoint runs/pilot/train/best.pt --calib runs/pilot/calib --precision fp16 --device-profile orin_nx --out runs/pilot/export
python scripts/06_replay.py --manifest runs/pilot/data/frames.jsonl --split val --engine runs/pilot/export/edge.plan --out runs/pilot/replay
python scripts/05_run_ablations.py --suite configs/eval/edge_baseline.yaml --pred runs/pilot/replay --labels runs/pilot/labels --out runs/pilot/eval
```

Script 00 tạo cả `frames.jsonl`, `split.json`, `rig.json` và thống kê thiếu modality. Script 01 xuất LUT theo rig/hash và ảnh overlay. Script 02 tạo manifest nhãn kèm shard và provenance. Script 03 ghi `last.pt` để resume, `best.pt` theo validation. Script 04 có chế độ export graph với fixture và chế độ build/bench trên thiết bị đích. Script 06 xuất prediction cùng `frame_id`, state validity và `events.jsonl`. Script 05 có chế độ chấm một run như ví dụ và chế độ chạy ma trận từ config. Một run đủ để kiểm tra lại phải có các ID đầu vào, dự đoán, nhãn, metric, phiên bản evaluator và số mẫu bị loại.

**Resume và lỗi.** Khóa ghi theo run, xuất shard/checkpoint vào tệp tạm rồi rename khi hoàn tất, đánh dấu shard hoàn thành sau checksum. Khi resume, kiểm hash config/rig/grid/label/split, khôi phục optimizer, scheduler, AMP scaler, EMA, RNG và vị trí sampler. Cache có hash khác được ghi bằng khóa mới. Script trả exit code khác 0 cho schema sai, LUT sai rig, split rò rỉ, engine sai binding hoặc lỗi parity. Mẫu thiếu một cảm biến vẫn đi theo đường xử lý degraded có ghi lý do. Báo cáo xuất `not_measured` cho phần cứng, nhãn hay benchmark chưa có số liệu.

**Bàn giao cho người nhận mốc đầu.** Mở một overlay từ M0, chọn một pixel/voxel nhìn thấy và một voxel khuất, lần ngược được timestamp, cảm biến, tọa độ, bin độ sâu và nguồn nhãn. Sau M3, dùng chính clip đó chạy lại PyTorch và TensorRT, đối chiếu tensor trung gian và đọc p95 từ log replay. Đây là phép thử tích hợp đầu tiên trước khi đầu tư vào lượt train nhiều giai đoạn.
