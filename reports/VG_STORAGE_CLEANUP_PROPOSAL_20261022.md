# Đề xuất dọn dung lượng —22/10/2026

Chỉ kiểm kê/preflight, chưa xóa hoặc di chuyển tệp. Lịch tự động vẫn PAUSED.

Phương án ưu tiên: xóa đúng480shards train/val `.pt` đã được liệt kê trong manifest của `~/rsg-former/artifacts/vg_method_v1/features/`, khoảng20.20GiB. V1 đã kết thúc âm tính `screening_failed_to_establish_benefit`, chưa chấm test và không được restart/chấm lại. Cache này chứa frozen ImageNet/GT-box features của V1, tách biệt cache detector/SGDet V2.

Giữ nguyên `features/manifest.json`, toàn bộ checkpoint, source/config, logs, validation records, outcome, dữ liệu/VG ZIP, journal/pHash, shared assets và các tệp dự đoán Context TEST. Xóa feature shards làm mất khả năng dùng ngay cache V1; muốn tạo lại phải tốn compute, không được tự động tạo lại hoặc restart V1.

Dung lượng trống đo mới nhất4.99GiB; sau dọn ước tính25.19GiB nếu không có thay đổi từ các job khác. Riêng pip download cache khoảng0.35GiB, có thể tái tải nhưng không giải quyết thiếu chỗ; chưa đưa vào phạm vi dọn ưu tiên. Cache V2 khoảng20.36GiB và bằng chứng prediction TEST được giữ.

Preflight kiểm tra tất cả tên/size/shard membership, không symlink, mỗi shard có1hardlink; giữ expectedSHA từ manifest. ManifestSHA `d39354aa61511c2bc3a02fa0b0115a1ad6473c2a48fb8ffcc4936043de95519e`. Chưa hash lại nội dung21.7GBshards. Không phát hiện consumer V1 trong các own process đọc được;3process checks không đọc đủ, tài khoản khác không kiểm tra. Khi được người dùng chốt xóa: kiểm tra mới consumer, paths/stat/manifest và bảo toàn evidence trước khi xóa đúng whitelist; không dùng recursive deletion toàn thư mục. Nếu có blocker/đổi tệp phải dừng.

Danh sách exact480tệp và device/inode/mtime/size/expectedSHA trong `reports/VG_V1_CACHE_CLEANUP_PREFLIGHT.json`. Cần người dùng xác nhận phạm vi này trước thao tác xóa.
