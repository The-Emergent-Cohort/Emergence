# Kaggle API & Platform Reference Guide

*Generated: December 2024 | For Emergence Project*

## Quick Summary

| Feature | Available | Method |
|---------|-----------|--------|
| Search/list datasets | Yes | `kaggle datasets list --search "term" --tags "tag"` |
| Download datasets | Yes | `kaggle datasets download -d owner/name` |
| Upload datasets | Yes | `kaggle datasets create -p /path` |
| Run notebooks w/ GPU | Yes | `kaggle kernels update` + metadata |
| Submit to competitions | Yes | `kaggle competitions submit` |
| Webhooks | No | Use GitHub Actions or Pipedream |
| Free GPU hours | 30hr/week | P100 or dual T4 |

---

## 1. Installation & Authentication

```bash
pip install kaggle kagglehub
```

**Setup:**
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New Token" → downloads `kaggle.json`
3. Place at `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<user>\.kaggle\kaggle.json` (Windows)

**Or use environment variables:**
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

---

## 2. Dataset Operations

### Search & List
```bash
# Search by keyword
kaggle datasets list --search "education adaptive learning"

# Filter by tags
kaggle datasets list --tags "machine-learning,classification"

# Filter by file type
kaggle datasets list --file-type "csv"

# Sort options: hottest, votes, updated, active
kaggle datasets list --sort-by "hottest"
```

### Download
```bash
# Download entire dataset
kaggle datasets download -d owner/dataset-name

# Download + unzip
kaggle datasets download -d owner/dataset-name --unzip

# Download specific file
kaggle datasets download -d owner/dataset-name -f filename.csv

# Specify path
kaggle datasets download -d owner/dataset-name -p ./data/
```

### Python API (KaggleHub - newer)
```python
import kagglehub

# Download dataset
path = kagglehub.dataset_download('owner/dataset-name')

# Download specific version
path = kagglehub.dataset_download('owner/dataset-name/versions/1')

# Download single file
path = kagglehub.dataset_download('owner/dataset-name', path='file.csv')
```

### Upload/Publish
```bash
# Initialize metadata
kaggle datasets init -p /path/to/dataset

# Create new dataset
kaggle datasets create -p /path/to/dataset

# Create new version
kaggle datasets version -m "version message" -p /path/to/dataset
```

---

## 3. Notebooks/Kernels (GPU Access)

### Push Notebook to Run on Kaggle GPU
```bash
# Initialize kernel metadata
kaggle kernels init -p /path/to/notebook

# Push and run
kaggle kernels update -p /path/to/notebook

# Check status
kaggle kernels status username/kernel-slug

# Download output
kaggle kernels output username/kernel-slug -p ./output/
```

### Enable GPU in `kernel-metadata.json`
```json
{
  "id": "username/kernel-name",
  "title": "My Kernel",
  "code_file": "notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["owner/dataset-name"]
}
```

---

## 4. Competitions

```bash
# List competitions
kaggle competitions list

# Download competition data
kaggle competitions download -c competition-name

# Submit predictions
kaggle competitions submit -c competition-name -f submission.csv -m "my submission"

# Check submissions
kaggle competitions submissions -c competition-name

# View leaderboard
kaggle competitions leaderboard -c competition-name
```

---

## 5. Rate Limits & Quotas (Free Tier)

| Resource | Limit |
|----------|-------|
| GPU hours/week | 30 hours (can increase to 40+ with activity) |
| TPU hours/week | 30 hours |
| Session time (GPU) | 12 hours max |
| Session time (TPU) | 9 hours max |
| Idle timeout | 20 minutes |
| Disk space | 20 GB |
| Daily submissions | ~5 per competition |

**GPU Options:**
- NVIDIA P100: 16GB VRAM
- Dual NVIDIA T4: 2x 16GB VRAM

**Note:** Must verify phone number to access GPU/TPU

---

## 6. Integration & Automation

### GitHub Actions (Scheduled/Triggered Runs)
```yaml
name: Kaggle Training
on:
  schedule:
    - cron: '0 9 * * *'  # Daily at 9 AM
  workflow_dispatch:  # Manual trigger

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Kaggle
        run: |
          pip install kaggle
          mkdir -p ~/.kaggle
          echo '${{ secrets.KAGGLE_JSON }}' > ~/.kaggle/kaggle.json
          chmod 600 ~/.kaggle/kaggle.json
      - name: Push Kernel
        run: kaggle kernels update -p ./notebook/
```

### Third-Party Integrations
- **Pipedream**: HTTP triggers → Kaggle API actions
- **n8n**: Webhook triggers + API nodes
- **GitHub Actions**: Scheduled or event-triggered

**No native webhooks** - must poll or use external triggers.

---

## 7. Useful Commands Reference

```bash
# Datasets
kaggle datasets list [--search TERM] [--tags TAGS] [--sort-by hottest|votes|updated]
kaggle datasets download -d OWNER/NAME [--unzip] [-p PATH]
kaggle datasets create -p PATH
kaggle datasets version -m "message" -p PATH

# Kernels
kaggle kernels list
kaggle kernels init -p PATH
kaggle kernels update -p PATH
kaggle kernels status OWNER/KERNEL
kaggle kernels output OWNER/KERNEL -p PATH

# Competitions
kaggle competitions list
kaggle competitions download -c COMPETITION
kaggle competitions submit -c COMPETITION -f FILE -m "message"
kaggle competitions leaderboard -c COMPETITION

# Config
kaggle config view
kaggle --version
```

---

## 8. Key Links

- **API Docs**: https://www.kaggle.com/docs/api
- **GitHub (kaggle-api)**: https://github.com/Kaggle/kaggle-api
- **GitHub (kagglehub)**: https://github.com/Kaggle/kagglehub
- **PyPI**: https://pypi.org/project/kaggle/
- **Education Datasets**: https://www.kaggle.com/datasets?tags=11105-Education

---

## 9. For Our Use Case (Emergence Project)

**Potential uses:**
1. **Dataset search**: Find education/learning datasets programmatically
2. **Remote GPU training**: Push notebooks to run on free P100/T4
3. **Model hosting**: Upload trained models for sharing
4. **Parallel experiments**: Run multiple configs on Kaggle while local machine does other work

**Workflow idea:**
```
Local: Develop & test small models
  ↓
Kaggle: Scale up training with GPU (30hr/week free)
  ↓
Local: Analyze results, iterate
```

**Search for education datasets:**
```bash
kaggle datasets list --search "student learning adaptive" --tags "education" --sort-by votes
kaggle datasets list --search "mastery curriculum progression" --file-type csv
```
