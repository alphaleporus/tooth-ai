# Local Setup for Windows (RTX 4060 / CUDA)

## Prerequisites

### 1. Install Git
```powershell
winget install --id Git.Git -e --source winget
```

### 2. Install Visual Studio Build Tools (Required for Detectron2)
```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --accept-package-agreements --accept-source-agreements --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows10SDK.19041"
```

> **Important**: You may need to restart your computer after installation for the changes to take effect.

---

## Setup Steps

### Step 1: Clone the Repository
```powershell
git clone https://github.com/alphaleporus/tooth-ai.git
cd tooth-ai
```

### Step 2: Create Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Core Dependencies
```powershell
pip install --upgrade pip wheel ninja
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python wandb numpy tqdm
pip install matplotlib seaborn scikit-learn pycocotools
```

### Step 4: Install Detectron2

Create a batch file `install_detectron2.bat` with:
```batch
@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=%PATH%;C:\Users\%USERNAME%\AppData\Local\Programs\Git\bin
set DISTUTILS_USE_SDK=1
pip install git+https://github.com/facebookresearch/detectron2.git --no-build-isolation
```

Then run:
```powershell
.\install_detectron2.bat
```

### Step 5: Login to WandB
```powershell
wandb login
# Enter your API key from https://wandb.ai/authorize
```

### Step 6: Run Local Test
```powershell
python workspace\tools\test_local.py
```

---

## Expected Test Output

```
[OK] PyTorch (2.x.x+cu121)
[OK] TorchVision
[OK] Detectron2 (0.6)
[OK] OpenCV
[OK] Weights & Biases
[OK] NumPy
[OK] tqdm
[OK] CUDA available
```

---

## Troubleshooting

### Detectron2 Build Fails with "Cannot open include file: 'assert.h'"
This means Windows SDK is not installed. Install it via:
1. Open **Visual Studio Installer**
2. Click **Modify** on Build Tools 2022
3. Under **Individual components**, check **Windows 10 SDK (10.0.19041.0)**
4. Click **Modify** and wait for installation

### WandB Login Issues
```powershell
wandb login --relogin
```

---

## Training Configuration

The config has been optimized for RTX 4060 (8GB VRAM):
- Batch size: 2
- Learning rate: 0.0005
- Iterations: 48,000
- Workers: 2

See `workspace/configs/mask_rcnn_1024x512.yaml` for full details.
