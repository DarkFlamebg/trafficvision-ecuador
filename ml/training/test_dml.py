import torch
try:
    import torch_directml
    dml = torch_directml.device()
    print("DirectML Device:", dml)
    tensor = torch.tensor([1.0, 2.0]).to(dml)
    print("Tensor on DirectML:", tensor)
    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
