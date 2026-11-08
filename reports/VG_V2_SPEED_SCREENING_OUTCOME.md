# Screening hoàn tất — 14/10/2026

Cả bốn nhánh seed1729 đã hoàn tất6epochs; tất cả selected epoch6. Validation5000 ảnh, SGDet GC.

| Nhánh | R@50 (%) | mR@50 (%) |
|---|---:|---:|
| native | 26.0278 | 16.6467 |
| context | 26.2568 | 16.2430 |
| oneway | 26.0996 | 16.6954 |
| feedback | 26.0309 | 16.8343 |

Feedback được chọn ở screening: gain mR@50 so native 0.1876pp, context 0.5913pp, oneway 0.1389pp. Chỉ một seed; chưa chứng minh lợi ích ổn định hoặc SOTA.

Confirmation chưa chạy: resource admission yêu cầu 642.39h cho156epochs và test allowance, còn 161.02h trong budget gốc tại thời điểm admission; thiếu 481.37h. Đây là guard dùng maximum observed epoch với25%margin, chưa phải cam kết thời gian. Không tăng cap hoặc mở test.

OI stream kết thúc với125225/125436 fingerprints,211ID hết lượt thử. HaiID có supplemental repair,209ID chưa có supplemental fingerprints; không giảm denominator. Coverage/perceptual và missing COCO/Flickr vẫn chưa hoàn tất. Caption-clean chưa được chứng nhận.

Tiếp tục một batch pHash pinned mỗi lượt12h và chuẩn bị canonical/caption độc lập trong caps. Không restart training cũ; quyết định confirmation hoặc phiên bản khoa học tiếp theo cần preregistration/admission mới.
