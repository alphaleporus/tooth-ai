# Tooth-AI Deployment Handover Guide

## Project Migration: Windows → Mac via GitHub

---

## Step 1: Install Git LFS

```powershell
# Download and install Git LFS (run once per machine)
winget install GitHub.GitLFS

# Initialize Git LFS in your repo
$env:Path = $env:Path + ";C:\Users\Student\AppData\Local\Programs\Git\bin"
git lfs install
```

---

## Step 2: Configure Git LFS for Large Files

```powershell
# Track .pth model files with LFS
git lfs track "*.pth"
git lfs track "output/**/*.pth"

# Verify .gitattributes was created
cat .gitattributes
```

Expected `.gitattributes` content:
```
*.pth filter=lfs diff=lfs merge=lfs -text
output/**/*.pth filter=lfs diff=lfs merge=lfs -text
```

---

## Step 3: Stage and Commit

```powershell
# Reset any previous staging
git reset HEAD

# Add specific files for deployment
git add .gitignore
git add .gitattributes
git add workspace/
git add SETUP_WINDOWS.md
git add install_detectron2.bat
git add verify_gpu.py

# Add the trained model (will use LFS)
git add output/rtx4060_48k/config.yaml
git add output/rtx4060_48k/model_final.pth

# Check status
git status

# Commit
git commit -m "Deployment: Add trained model (48K iters, 16% mAP) via Git LFS"
```

---

## Step 4: Push to GitHub

```powershell
git push origin main
```

If push fails due to file size, verify LFS is tracking:
```powershell
git lfs ls-files
```

---

## Step 5: Clone on Mac

```bash
# Clone with LFS
git lfs install
git clone https://github.com/alphaleporus/tooth-ai.git
cd tooth-ai

# Verify model downloaded
ls -la output/rtx4060_48k/model_final.pth
```

---

## Files Included in Deployment

| Path | Size | Description |
|------|------|-------------|
| `workspace/` | ~50KB | Training scripts & configs |
| `output/rtx4060_48k/config.yaml` | ~15KB | Model architecture config |
| `output/rtx4060_48k/model_final.pth` | ~170MB | Trained weights (LFS) |
| `.gitignore` | ~1KB | Exclusion rules |
| `SETUP_WINDOWS.md` | ~3KB | Setup documentation |

---

## Troubleshooting

### "File too large" error
```powershell
git lfs migrate import --include="*.pth" --everything
```

### LFS not installed on Mac
```bash
brew install git-lfs
git lfs install
```
