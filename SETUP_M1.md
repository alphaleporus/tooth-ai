# Local Setup for MacBook M1

## Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install PyTorch for M1
pip install torch torchvision torchaudio

# Install Detectron2 (M1 compatible)
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install other dependencies
pip install opencv-python wandb numpy tqdm
```

## Step 2: Login to WandB

```bash
wandb login
# Enter your API key from https://wandb.ai/authorize
```

## Step 3: Run Local Test

```bash
python workspace/tools/test_local.py
```

## Step 4: Quick Training Test (CPU, ~5 min)

```bash
python workspace/train_net.py \
  --config-file workspace/configs/mask_rcnn_1024x512.yaml \
  MODEL.DEVICE cpu \
  SOLVER.MAX_ITER 50 \
  SOLVER.IMS_PER_BATCH 1
```

## Notes for M1

- Detectron2 works on M1 but training is SLOW (CPU only, no CUDA)
- Use this only for testing, not full training
- For full training, use RunPod with GPU

## Expected Test Output

```
✓ PyTorch
✓ Detectron2
✓ OpenCV
✓ WandB logged in
✓ Dataset: 6,992 images
✓ Config: 41 classes
🎉 All checks passed!
```
