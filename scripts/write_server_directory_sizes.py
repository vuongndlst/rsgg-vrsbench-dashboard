"""Render and verify the read-only server storage inventory."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv, json

root = Path(__file__).absolute().parents[1]
report = json.loads((root/'reports/SERVER_DIRECTORY_SIZES_FULL.json').read_text(encoding='utf8'))
summary = json.loads((root/'reports/SERVER_DIRECTORY_SIZES_SUMMARY.json').read_text(encoding='utf8'))
assert report['home_top_level_entries'] == summary['home_top_level_entries']
with (root/'reports/SERVER_DIRECTORY_SIZES_ALL.csv').open(encoding='utf-8-sig', newline='') as handle:
    csv_rows = list(csv.DictReader(handle))
expected = {(scope, row['path']): row['allocated_bytes']
            for scope, name in [('root_depth1','root_directories'), ('home_recursive','home_all_directories')]
            for row in report[name]['rows']}
assert len(csv_rows) == len(expected)
assert all(expected[(row['scope'],row['path'])] == int(row['allocated_bytes']) for row in csv_rows)
time_vn = datetime.fromisoformat(report['utc']).astimezone(timezone(timedelta(hours=7)))
capacity = report['home_capacity_after']
def size(amount):
    if amount is None: return 'Không đọc được'
    if amount >= 2**30: return f'{amount/2**30:.2f} GiB'
    if amount >= 2**20: return f'{amount/2**20:.2f} MiB'
    return f'{amount/2**10:.0f} KiB'
home_rows = '\n'.join('| `' + row['path'].replace('${REMOTE_HOME}/','~/') + '` | ' + size(row.get('allocated_bytes')) + ' |'
                      for row in report['home_top_level_entries'])
system_rows = '\n'.join('| `' + row['path'] + '` | ' + size(row['allocated_bytes']) + ' |'
                        for row in report['root_directories']['rows'] if row['path'] != '/')
large_rows = '\n'.join('| `' + row['path'].replace('${REMOTE_HOME}/','~/') + '` | ' + size(row['allocated_bytes']) + ' |'
                      for row in summary['largest_home_descendants'][:20])
content = f'''# Dung lượng server — {time_vn:%d/%m/%Y %H:%M:%S} giờ Việt Nam

Ổ ext4 dùng chung `/`: tổng **{size(capacity['total_bytes'])}**, đã dùng **{size(capacity['used_bytes'])}**, khả dụng cho tài khoản **{size(capacity['available_bytes'])}**. Tổng block còn trống là {size(capacity['free_bytes'])}; trong đó {size(capacity['reserved_or_unavailable_bytes'])} là phần dự trữ/không khả dụng cho tài khoản.

Home chiếm **{size(summary['home_total_allocated_bytes'])}**. Quét đủ **{summary['home_directory_count']:,} thư mục vật lý**, gồm thư mục ẩn và thư mục con: không báo lỗi. Đo trong {report['wall_seconds']:.1f} giây, ưu tiên CPU/I/O thấp, một CPU. Đây là snapshot; các job vẫn có thể ghi thêm dữ liệu. Không xóa tài sản hay thay đổi job.

[CSV đầy đủ — thư mục, byte và GiB](/D:/SBIR/RSGG/reports/SERVER_DIRECTORY_SIZES_ALL.csv)

## Tất cả thư mục cấp đầu trong home

`~` là home của tài khoản SSH. Dung lượng cha bao gồm mọi thư mục con, không cộng các dòng cha/con với nhau. Số đo dùng block thực tế `du`, không chạy theo symlink và chỉ tính hard link một lần trong mỗi lượt quét.

| Thư mục | Dung lượng |
|---|---:|
{home_rows}

## Thư mục cấp đầu ở `/`

Quét chỉ đọc ở depth1, có {report['root_directories']['stderr_lines']} dòng lỗi truy cập; các số có thể thiếu phần không có quyền đọc. Không dùng sudo. Các filesystem ảo `/proc`, `/sys`, `/dev`, `/run` được loại khỏi đo dung lượng đĩa.

| Thư mục | Dung lượng đọc được |
|---|---:|
{system_rows}

## Các thư mục con lớn nhất

Các dòng lồng nhau đã nằm trong dung lượng cha; không cộng lại.

| Thư mục | Dung lượng |
|---|---:|
{large_rows}

`df` bao gồm toàn filesystem; `du` chỉ tính phần đọc được. Permission, hard link, file đã xóa nhưng còn được process mở và dữ liệu thay đổi trong lúc quét có thể tạo chênh lệch. Lượt này không kiểm tra hay kết luận loại nào gây ra toàn bộ chênh lệch.
'''
# Use native absolute Windows links in the saved Markdown.
content=content.replace('/D:/SBIR/RSGG/', 'D:/SBIR/RSGG/')
destination=root/'reports/SERVER_DIRECTORY_SIZES.md'
destination.write_text(content, encoding='utf8')
print(json.dumps({'utc':report['utc'],'time_VN':time_vn.isoformat(),'available_GiB':round(capacity['available_bytes']/2**30,2),
                  'total_GiB':round(capacity['total_bytes']/2**30,2),'used_GiB':round(capacity['used_bytes']/2**30,2),
                  'home_GiB':round(summary['home_total_allocated_bytes']/2**30,2),
                  'top_level_home_directories':len(report['home_top_level_entries']),
                  'all_home_directories':summary['home_directory_count'],'csv_semantic_check':'PASS',
                  'report':str(destination)}))
