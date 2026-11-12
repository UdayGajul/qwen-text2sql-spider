# Text-to-SQL: Fine-Tuning Qwen2.5-Coder-7B on Spider

Fine-tuned **Qwen2.5-Coder-7B-Instruct** with LoRA (QLoRA, 4-bit NF4) on the [Spider benchmark](https://yale-lily.github.io/spider), achieving **77.7% execution accuracy** — outperforming a zero-shot Llama 3.3-70B baseline by **+24.5%**.

**🤗 HuggingFace Adapter:** [Gansaw98/qwen2.5-coder-7b-text2sql-spider](https://huggingface.co/Gansaw98/qwen2.5-coder-7b-text2sql-spider) | **🚀 Live Demo:** [text-to-sql-demo](https://huggingface.co/spaces/Gansaw98/text-to-sql-demo)

---

## Results

| Model | Approach | Execution Accuracy | Exact Match |
|-------|----------|:-----------------:|:-----------:|
| **Qwen2.5-Coder-7B (ours)** | LoRA fine-tuned | **77.7%** | **77.4%** |
| Llama 3.3-70B (Groq) | Zero-shot | 53.2% | 46.0% |

Evaluated on the Spider dev set (1,034 examples) using the official [Yale-LILY evaluation script](https://github.com/taoyds/spider).

### By difficulty

| Difficulty | Fine-tuned 7B | Groq 70B (zero-shot) |
|------------|:------------:|:-------------------:|
| Easy | 89.1% | 85.1% |
| Medium | 82.1% | 48.7% |
| Hard | 69.5% | 47.1% |
| Extra | 57.2% | 24.1% |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

For Google Colab with a T4 GPU, use the reproducible training dependencies:

```bash
pip install -r requirements-colab.txt
```

### 2. Download Spider dataset
Download Spider 1.0 from the [official Yale-LILY Spider page](https://yale-lily.github.io/spider)
and extract it into `data/spider/`. The current official Google Drive file is
linked from that page; older file IDs may no longer work.

Expected structure:
```
data/spider/
├── dev.json
├── dev_gold.sql
├── tables.json
└── database/       # SQLite files
```

---

## Training

Training was done on a **T4 GPU** (free tier). The original notebook is in
`training/text-to-sql.ipynb`; it targets Kaggle. For Colab, use
`training/train_colab.py`, which removes Kaggle-only secrets and paths.

Example from the repository root in Colab:

```bash
mkdir -p /content/spider-download
# Download spider.zip from the official page, then upload it to:
# /content/spider-download/spider.zip
python training/prepare_spider.py \
  /content/spider-download/spider.zip \
  --output data/spider
python training/train_colab.py \
  --tables data/spider/tables.json \
  --output-dir /content/qwen-text2sql-adapter
```

For a clean Colab runtime, clone this repository instead of extracting an old
ZIP copy:

```python
!git clone https://github.com/UdayGajul/qwen-text2sql-spider.git
%cd /content/qwen-text2sql-spider
!pip install -r requirements-colab.txt
```

To evaluate the adapter you trained instead of the published adapter, set:

```bash
export HF_REPO_ID=/content/qwen-text2sql-adapter
```

The published adapter can be evaluated without retraining, which is the
recommended path when time is limited:

```bash
export SPIDER_ROOT=/content/text-to-sql-spider/data/spider
python evaluation/finetuned_eval.py --limit 50
```

After a full run, replace `--limit 50` with no limit and run the official
Spider scorer:

```bash
python evaluation/evaluation.py \
  --gold data/spider/dev_gold.sql \
  --pred results/finetuned_predicted.sql \
  --db data/spider/database \
  --table data/spider/tables.json \
  --etype all
```

**Key hyperparameters:**
- Base model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- LoRA rank: 16, alpha: 32
- Quantization: 4-bit NF4 (QLoRA)
- Trainable params: 40.4M (0.6% of total)
- Epochs: 2, LR: 2e-4 (cosine)
- Training examples: 6,836 (filtered at 1024 tokens)

---

## Evaluation

### Fine-tuned model (run on Colab or another CUDA GPU)
```bash
python evaluation/finetuned_eval.py
```

### Groq baseline (Llama 3.3-70B zero-shot)
```bash
# Requires GROQ_API_KEY in .env
python evaluation/groq_baseline.py
```

### Official Yale-LILY scorer (run locally with SQLite DBs)
```bash
python evaluation/evaluation.py \
  --gold data/spider/dev_gold.sql \
  --pred results/finetuned_predicted.sql \
  --db data/spider/database \
  --table data/spider/tables.json \
  --etype all
```

### Compare both models
```bash
python evaluation/metrics.py
```

---

## Project Structure

```
├── evaluation/
│   ├── finetuned_eval.py      # Fine-tuned model inference + eval loop
│   ├── groq_baseline.py       # Groq API baseline
│   ├── schema_utils.py        # Schema prompt builder (shared with training)
│   ├── sql_executor.py        # SQLite execution for quick EX checks
│   ├── metrics.py             # Side-by-side comparison table
│   ├── evaluation.py          # Yale-LILY official evaluation script
│   └── process_sql.py         # Yale-LILY SQL parser
├── training/
│   ├── text-to-sql.ipynb      # Full training notebook (Kaggle)
│   └── _measure_lengths.py    # Token length analysis
├── deployment/
│   └── app.py                 # Gradio demo (HF Spaces)
├── results/
│   ├── finetuned_predicted.sql
│   └── groq_predicted.sql
├── .env.example
└── requirements.txt
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:
```
GROQ_API_KEY=your_groq_key      # For groq_baseline.py
HF_TOKEN=your_hf_token          # For pushing adapter to HF Hub
```
