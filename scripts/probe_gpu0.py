import os, torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')
print("Testing CUDA allocation on GPU 0...")
try:
    x = torch.empty((1, 3, 1024, 1024), dtype=torch.bfloat16, device=device)
    print("Allocated 1x3x1024x1024 bfloat16 successfully!")
    y = torch.empty((2048, 2048, 128), dtype=torch.float32, device=device)
    print("Allocated 2GB tensor successfully!")
    del x, y
    torch.cuda.empty_cache()
    print("GPU 0 probe SUCCESSFUL! Free memory:", torch.cuda.mem_get_info())
except Exception as e:
    print("GPU 0 probe failed:", e)

