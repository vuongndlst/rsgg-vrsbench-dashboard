# -*- coding: utf-8 -*-
"""Use the unchanged display selection on the six saved external predictions."""
from pathlib import Path
root=Path(__file__).absolute().parents[1]
code=(root/'scripts/render_context_test_visualizations.py').read_text(encoding='utf8')
code=code.replace('context_1729_test_visualization_20261024','context_1729_coco_oi_visualization_20261024')
code=code.replace('VG TEST  /  IMAGE','EXTERNAL TEST  /  IMAGE').replace(' /  VG {identity}',' /  {identity}')
code=code.replace('held-out test sample  |  saved model outputs','external qualitative sample  |  VG150 output vocabulary')
code=code.replace('SIX FIXED VG TEST SAMPLES','3 COCO + 3 OPEN IMAGES TEST SAMPLES')
code=code.replace('test index','sample order').replace('VG {g["image_id"]}','{g["image_id"]}')
code=code.replace('Dùng lại dự đoán của lượt test đã hoàn tất, không chạy model hay chấm metric lại.','Suy luận mới trên 3 ảnh COCO Karpathy test và 3 ảnh Open Images test; không chấm metric. Giữ nguyên checkpoint và từ vựng 150 đối tượng/50 quan hệ của VG.')
code=code.replace('Sáu ảnh được chọn cố định bằng random seed 1729 từ VG test; không chọn theo chất lượng dự đoán.','Chọn cố định bằng random seed 1729 từ các ảnh test sẵn có: COCO 5000/5000; Open Images chỉ có 128 ảnh local trong 125436 ID. Không chọn theo chất lượng dự đoán; đây không phải đánh giá đại diện toàn bộ dữ liệu.')
code=code.replace('No new inference, evaluator or test metric was run.','New frozen-checkpoint inference on6fixed external images; no evaluator, test metric, calibration or training. VG150 output vocabulary; not an external benchmark or caption-clean generalization claim. Server files written:0. Storage-blocked preparation remains paused.')
code=code.replace('Original images were read directly from existing server ZIPs','Original images were read directly from existing shared-assets files')
code=code.replace("'new_inference':False","'new_inference':True")
exec(compile(code,str(root/'scripts/render_context_test_visualizations.py'),'exec'),{'__file__':str(root/'scripts/render_context_test_visualizations.py'),'__name__':'__main__'})
