# Dọn cache V1 và tiếp tục —22/10/2026

Người dùng đã xác nhận: “ok dọn và tiếp tục”. Đã xóa đúng480tệp feature shards `.pt` của thí nghiệm V1 kết thúc âm tính. Không xóa recursive cả thư mục, không restart V1 hoặc chấm lại test.

- Dung lượng thu hồi theo allocated files: 20.20GiB; free-space gain quan sát: 20.20GiB.
- Dung lượng trống sau dọn: 25.19GiB tại `2026-10-21T22:31:03.091400+00:00`, vượt reserve16GiB+128MiB. Các job khác trên shared server có thể làm thay đổi dung lượng sau snapshot.
- Giữ manifest gốc SHA `d39354aa61511c2bc3a02fa0b0115a1ad6473c2a48fb8ffcc4936043de95519e`; giữ checkpoint/source/config/logs/metrics của V1, cache V2, bằng chứng và predictions Context TEST, journals/ảnh và shared assets.
- So sánh device/inode/size/mtime của 2472tệp còn lại trước/sau dọn đạt; không có ứng dụng đang mở/map cache trong own processes đọc được. PAM/SSH/SFTP platform services được xác minh riêng, không bị dừng; tài khoản khác không kiểm tra.
- Audit giữ tại `artifacts/maintenance/vg_v1_cache_retirement_20261022`: original manifest, exact whitelist/pre-delete protected signatures, deletion journal và result. Cache V1 đã retire có chủ đích; old cache-integrity smoke không thể chạy trên shards đã xóa. Muốn tạo lại phải tốn compute, không được tự động tái tạo.

Tiếp tục đúng một batch pHash pinned để xác nhận storage admission và khôi phục lịch12h sau khi batch thành công. Không chạy bù nhiều batch, sửa phương pháp/journal hoặc chấm lại Context TEST.
