import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Compute capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "n/a")
# The real test — run an actual computation on the GPU:
x = torch.rand(2000, 2000).cuda()
y = (x @ x).sum()
print("GPU compute works:", y.item() > 0)