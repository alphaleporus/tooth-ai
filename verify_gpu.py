import torch
import detectron2
from detectron2.utils.collect_env import collect_env_info

print("---------------------------------------")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
print(f"Detectron2 Ver:  {detectron2.__version__}")
print("---------------------------------------")
