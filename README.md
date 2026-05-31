# deep-learning-NLP

This repository currently tracks one runnable project:

- AG News topic classification in `01_tensor_basics/AG-news-classification`

## How to run after cloning

The tracked code supports:

- Data health check
- Training a model from CSV data
- Evaluation
- Inference

## 1) Clone

```bash
git clone https://github.com/<your-username>/deep-learning-NLP.git
cd deep-learning-NLP
```

Replace `<your-username>` with the real GitHub username.

## 2) Create and activate a Python virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3) Install dependencies

```bash
pip install -r 01_tensor_basics/AG-news-classification/requirments.txt
```

Notes:

- The file is named `requirments.txt` in this repo.
- If CUDA wheels fail to install, install CPU-only PyTorch from https://pytorch.org/get-started/locally/ and then rerun the command above.

## 4) Add required data files (important)

After clone, place these CSV files in the data folder:

- `01_tensor_basics/AG-news-classification/data/news_train_dataset.csv`
- `01_tensor_basics/AG-news-classification/data/news_test_dataset.csv`

If the `data` folder does not exist yet, create it first.

## 5) Run the AG News project

Run from the `src` directory so imports resolve correctly:

```bash
cd 01_tensor_basics/AG-news-classification/src
python -m news_topic_classification.util.data_health
python -m news_topic_classification.train
python -m news_topic_classification.eval
python -m news_topic_classification.infer
cd ../../..
```

Expected outputs:

- Trained model at `01_tensor_basics/AG-news-classification/models/news_topic_classifier.pt`
- Logs in `01_tensor_basics/AG-news-classification/logs/`

## Common issues

- `ModuleNotFoundError: No module named news_topic_classification`
  - You are not running from `01_tensor_basics/AG-news-classification/src`.

- `FileNotFoundError` for CSV files
  - Ensure both required files are present in `01_tensor_basics/AG-news-classification/data/`.

- Windows activation blocked in PowerShell
  - Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
  - Then run `.\.venv\Scripts\Activate.ps1`
