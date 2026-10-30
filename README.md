<<<<<<< HEAD
# Bangladesh Crop Yield Prediction

A local research prototype with a PHP website and a persistent FastAPI service. The backend uses the supplied CSV and the TabFM and TabPFN APIs already used by `crop-yield.ipynb`. It computes live metrics from actual held-out predictions. A model that is not installed, licensed, or loadable is reported as unavailable; there are no fallback models or sample scores.

## Data and research limits

The supplied `bangladesh_crop_yield_synthetic_1000.csv` is the source dataset. It contains 1,000 rows, 18 columns, target `crop_yield_tonnes_ha`, and `data_status=SYNTHETIC_FOR_PROTOTYPE`. The app excludes `record_id`, `country`, and `data_status`; uses the other 14 columns; imputes numeric missing values from training rows only; and evaluates a reproducible 80/20 split with random state 42. Both models see the same split. These results are software demonstrations on synthetic data, not validated estimates for Bangladesh farms.

The previous-season yield feature is kept to match the notebook. Enter it only if that information would genuinely be available at prediction time. Do not enter a value derived from the season being predicted.

## Requirements

- Windows 11 and Python 3.11 or newer (TabFM requires Python >=3.11; Python 3.12 is suitable).
- Git, PHP 8+, and XAMPP (Apache/PHP) or PHP's built-in server.
- A supported PyTorch CPU or CUDA build. GPU is optional; the backend selects CUDA when available.
- Local model weights and license/access acceptance for the exact TabFM and TabPFN model versions you use.

The backend does not silently install packages, clone repositories, prompt for tokens, or download weights at request time.

## Setup (PowerShell)

Open PowerShell in this project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch using the command recommended by the official [PyTorch install selector](https://pytorch.org/get-started/locally/) for your machine (CPU or the CUDA version installed). Then install the service and TabPFN:

```powershell
python -m pip install -r python_backend\requirements.txt
```

### TabFM

The TabFM repository is not included in this project. Clone the official repository beside this README and install its PyTorch extra in the same virtual environment:

```powershell
git clone https://github.com/google-research/tabfm.git
Set-Location tabfm
python -m pip install -e ".[pytorch]"
Set-Location ..
```

The backend first uses the official current imports (`tabfm.TabFMRegressor` and `tabfm.tabfm_v1_0_0_pytorch`) and then tries the import paths in the notebook. The pretrained TabFM v1.0.0 weights are downloaded by the official loader on first model initialization. **Those default weights have a separate non-commercial, non-production license**, distinct from the Apache-2.0 source license. Review and accept that license through the official model-weight distribution before use. A local license or Hugging Face access error will appear in the TabFM status and server log.

### TabPFN access

The notebook uses local `tabpfn.TabPFNRegressor(device=...)`, not the separate cloud client. Install TabPFN via the requirements file. Current TabPFN model releases require the user to accept the relevant model license and obtain/configure access as described by [Prior Labs](https://github.com/PriorLabs/TabPFN). Set `TABPFN_TOKEN` in the PowerShell session before starting the API if the selected local model version requests it:

```powershell
$env:TABPFN_TOKEN = "your-token"
```

Do not put a real token in source files or commit it. The PHP page never receives this value. Model-version licenses change; confirm that your intended research use is allowed by the accepted version's license. The TabPFN package may download checkpoint files the first time it initializes.

### Dataset

The supplied CSV is already in the project root. Keep the filename unchanged. To use a different path, set `CROP_YIELD_DATA` before starting the service:

```powershell
$env:CROP_YIELD_DATA = "C:\path\to\your\observed-or-synthetic-data.csv"
```

Missing data is reported as a service error; the backend never creates substitute data.

## Run

Start the Python API from the project root:

```powershell
python -m uvicorn main:app --app-dir python_backend --host 127.0.0.1 --port 8000
```

The service loads data, initializes the two models independently, and computes metrics at startup. Initial startup may take several minutes while packages/checkpoints load. Watch the console for each model's readiness. Open `http://127.0.0.1:8000/health` to inspect service and model status, `http://127.0.0.1:8000/metrics` for scores, and `http://127.0.0.1:8000/docs` for the local API schema.

On CPU, a prediction using both models can also take several minutes. The XAMPP form allows up to six minutes for a submission and the PHP-to-API request allows up to five minutes; keep the Python backend running while the request completes.

In a second PowerShell window, from this project root, start the PHP site:

```powershell
php -S 127.0.0.1:8080
```

Open `http://127.0.0.1:8080`. For XAMPP, place this project folder under `C:\xampp\htdocs\` and use `http://localhost/Crop%20yield/` (or your chosen folder name); the Python API still runs separately on port 8000. PHP's `allow_url_fopen` must be enabled for the built-in API client.

## Endpoints

- `GET /health`: dataset and per-model readiness.
- `GET /metadata`: features, dataset-derived category lists, and evaluation protocol.
- `GET /metrics`: real MSE, RMSE, MAE, and R² for models that completed held-out predictions.
- `POST /predict`: independently returns valid TabFM/TabPFN predictions, with an error status for a model that fails on that request.

Metrics are for one fixed holdout, not a repeated-split mean or cross-validation. Lower MSE, RMSE, and MAE indicate smaller errors. MSE is in squared tonnes/ha; RMSE and MAE are in tonnes/ha. R² measures improvement relative to predicting the test-set mean and can be negative. ROC AUC does not apply to these continuous regression predictions.

## Troubleshooting

- **API says the dataset is missing:** put the provided CSV in the project root or set `CROP_YIELD_DATA` to its full path.
- **TabFM is unavailable:** confirm the official repository is cloned to `<project>\tabfm`, installed in the active environment, and its checkpoint license/access is available. Inspect the API terminal for the original exception.
- **TabPFN is unavailable:** install the local `tabpfn` package, accept the relevant model version's license, configure `TABPFN_TOKEN` if required, and inspect the API terminal.
- **PHP reports the service is not responding:** start the API on `127.0.0.1:8000`; check Windows Firewall and the PHP `allow_url_fopen` setting.
- **CUDA out-of-memory:** start with a CPU-only PyTorch environment or reduce other GPU workloads. The current implementation chooses CUDA when PyTorch reports it available; see the limitation below.
- **A model is unavailable:** the site can still show the other model's actual result and held-out metrics.

## Current verification and limitations

The supplied notebook remains unchanged. The service's held-out model metrics and predictions are only produced when the official model packages and required checkpoint access are configured. This machine did not have TabFM, TabPFN, or FastAPI installed at the time the project was audited, so live model scores cannot be claimed until those dependencies and model access are set up and the service is run. The current backend records initialization failures independently. CUDA startup fallback is not automatic; select a CPU-only PyTorch environment if GPU initialization fails.
=======
# crop-yield-bd
>>>>>>> 038a5790e5f2505c52218b24bc775d611d0190c0
