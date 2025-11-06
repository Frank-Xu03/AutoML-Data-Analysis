# AutoML-LLM Data Analysis Platform

A practical data analysis platform that blends Automated ML (AutoML) with Large Language Models (LLMs).

中文版本: README.md

## What this project does

AutoML-LLM streamlines data workflows from ingestion and cleaning to feature engineering, model selection, and evaluation. LLMs add natural-language assistance for task detection and research question suggestions. You can use a friendly Streamlit UI or call the core modules programmatically.

## Key features

- Task detection powered by LLMs (or offline heuristics)
- Automated data cleaning and preprocessing
- Model selection and tuning via RandomizedSearchCV
- Streamlit-based visual UI
- AI-generated research question suggestions
- Artifacts and leaderboard export

## Project structure

```
AutoML Data Analysis/
├── automl-llm/
│   ├── app/
│   │   ├── ui_streamlit.py   # Streamlit UI
│   │   └── llm_agent.py      # LLM agent (OpenAI or offline)
│   ├── core/
│   │   ├── ingest.py         # Read/profile data
│   │   ├── cleandata.py      # Cleaning & preprocessing
│   │   ├── models.py         # Model registry and search spaces
│   │   └── train.py          # Training loop and leaderboard
│   └── artifacts/
├── examples/                  # Example CSVs
├── demos/                     # Demo scripts
├── docs/                      # Reports and notes
├── tests/                     # Test scripts
├── requirements.txt
├── usage_example.py
└── README.md / README_en.md
```

## Quick start

### 1) Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Optional: Configure OpenAI/compatible API

```powershell
# Create .env and add your key if you want online LLM
# (offline mode works without a key)
"OPENAI_API_KEY=your_api_key_here" | Out-File -FilePath .env -Encoding ascii
```

See docs/OPENAI_SETUP.md for details.

### 4) Run the app

```powershell
# Streamlit UI
streamlit run automl-llm/app/ui_streamlit.py

# or run the example script
python usage_example.py
```

## Environment variables and offline mode

You can run fully offline. When no API key is set or offline mode is enabled, the system falls back to robust heuristics.

- OPENAI_API_KEY: API key for OpenAI (optional)
- OPENAI_BASE_URL: Base URL for OpenAI-compatible services (optional)
- LLM_OFFLINE: Set to 1 to force offline mode (default 0)
- LLM_TASK_MODEL: Model name for task detection (default gpt-4o-mini)
- LLM_RESEARCH_MODEL: Model name for research suggestions (default gpt-4o-mini)
- LLM_TEMPERATURE: Sampling temperature (default 0.2)

PowerShell (current session only):

```powershell
$env:LLM_OFFLINE = "1"
$env:OPENAI_API_KEY = "sk-..."
```

Persist for new terminals:

```powershell
setx LLM_OFFLINE "1"
setx OPENAI_API_KEY "sk-..."
```

## How to use

### Web UI

1) Upload one or more CSV files
2) Inspect data profile and preview
3) Use "Discover Research Questions" to generate analysis ideas
4) Use "Detect Task" to get task type, candidate targets, algorithms, and metrics
5) Configure training and run models; view leaderboard and download results

CSV merge capabilities:
- Compute common columns across uploads
- Vertical stack: keep only common columns; optionally add `_source_file`
- Horizontal join: use common columns as keys; auto-prefix non-key columns to avoid collisions
- Merged results are saved in `examples/` as `merged_common.csv` or `merged_horizontal.csv`

### Programmatic usage

See `usage_example.py`. Typical flow:

```python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'automl-llm'))
from core.ingest import read_table
from core.cleandata import prepare
from core.train import run_all

# Load a CSV from examples/
df = read_table('tags.csv')

# Prepare
X_train, X_test, y_train, y_test, pre, col_info = prepare(df, target='tag', task_type='classification')

# Train a few models
leaderboard, artifacts = run_all(
    X_train, y_train, X_test, y_test,
    task_type='classification',
    picked_models=['rf','xgb'],
    preprocessor=pre,
    n_iter=30,
    cv_folds=5,
    artifacts_dir='artifacts'
)
```

## Tests

```powershell
python tests/test_complete_system.py
python tests/test_openai_setup.py
python tests/test_research_questions_fixed.py
python tests/test_ui_attributeerror_fix.py
```

Note: LLM-related tests will fall back to offline mode when no key is configured.

## Docs

- docs/FOLDER_ORGANIZATION_README.md
- docs/ATTRIBUTEERROR_FIX_REPORT.md
- docs/UI_IMPROVEMENT_REPORT.md
- docs/RESEARCH_QUESTIONS_FEATURE_REPORT.md
- docs/PYLANCE_FIX_REPORT.md

## Example data and artifacts

- Example CSVs in `examples/`: `movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`, etc.
- Training outputs under `artifacts/`, e.g. `leaderboard.csv`.

## Use cases

- Academia: quick EDA and hypothesis validation
- Enterprise: predictive modeling and decision support
- Education: ML & data science teaching tool
- Prototyping: build data-driven apps fast

## Contributing

PRs and issues are welcome:

1. Fork the repo
2. Create a branch: `git checkout -b feature/AmazingFeature`
3. Commit: `git commit -m "Add AmazingFeature"`
4. Push: `git push origin feature/AmazingFeature`
5. Open a PR

## Changelog (highlights)

- Fixed AttributeError cases
- Improved Streamlit UX
- Added research question suggestions
- Refined folder structure and tests

## Contact

- Issues: https://github.com/Frank-Xu03/AutoML-Data-Analysis/issues
- Docs Wiki: https://github.com/Frank-Xu03/AutoML-Data-Analysis/wiki

---

AutoML-LLM — make data analysis intelligent and efficient.
