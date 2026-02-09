# Quickstart (Windows / Mac / Linux)

## 1) Unzip
Extract the ZIP into a normal folder (not inside Downloads preview).

## 2) Open terminal in the folder
### Windows (easy)
- Open the extracted folder in File Explorer
- Click the address bar, type `cmd`, press Enter

### Mac/Linux
- Right-click folder → “Open in Terminal”

## 3) Create a virtual environment (recommended)
### Windows
```bat
py -m venv .venv
.venv\Scripts\activate
```

### Mac/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4) Install
```bash
pip install -r requirements.txt
```

## 5) Train + Infer
```bash
python run.py train
python run.py infer --ckpt runs/contextflow_ckpt.pt --topk 5
```

## If it still fails
- confirm Python is 3.10+ (`python --version`)
- confirm torch installed (`python -c "import torch; print(torch.__version__)"`)
