import torch

try:
    import torch_directml
    dml = torch_directml.device()
except:
    dml = torch.device('cpu')

_original_unique = torch.unique

def directml_safe_unique(input, *args, **kwargs):
    if input.device.type == 'privateuseone' and kwargs.get('return_counts', False):
        cpu_input = input.cpu()
        result = _original_unique(cpu_input, *args, **kwargs)
        if isinstance(result, tuple):
            return tuple(r.to(input.device) if isinstance(r, torch.Tensor) else r for r in result)
        return result.to(input.device) if isinstance(result, torch.Tensor) else result
    return _original_unique(input, *args, **kwargs)

torch.unique = directml_safe_unique

t = torch.tensor([1, 1, 2, 2, 3]).to(dml)
res = t.unique(return_counts=True)
print("Result:", res)
