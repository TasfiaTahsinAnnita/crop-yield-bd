# Bangladesh Crop Yield Prediction

A local research prototype with a PHP website and a persistent FastAPI service. The backend uses the supplied CSV for form categories and loads the pre-trained TabFM and TabPFN estimators from `crop_yield_models`. It does not fit or fine-tune models at startup. The supplied artifacts were trained on all 1,000 available synthetic rows, so the service does not report held-out metrics for them.

## Data and research limits

The supplied `bangladesh_crop_yield_synthetic_1000.csv` is the source dataset. It contains 1,000 rows, 18 columns, target `crop_yield_tonnes_ha`, and `data_status=SYNTHETIC_FOR_PROTOTYPE`. The app excludes `record_id`, `country`, and `data_status`, and uses the other 14 columns. It loads the saved numeric imputer and model artifacts, whose metadata lists 1,000 training rows. These models are software demonstrations on synthetic data, not validated estimates for Bangladesh farms.

The previous-season yield feature is kept to match the notebook. Enter it only if that information would genuinely be available at prediction time. Do not enter a value derived from the season being predicted.

## Requirements

- Windows 11 and Python 3.11 or newer (TabFM requires Python >=3.11; Python 3.12 is suitable).
- Git, PHP 8+, and XAMPP (Apache/PHP) or PHP's built-in server.
- A supported PyTorch CPU or CUDA build. GPU is optional; the backend selects CUDA when available.
- The model artifacts in `crop_yield_models` and license/access acceptance for the exact TabFM and TabPFN model versions they contain.

The backend does not silently install packages, clone repositories, prompt for tokens, train models, or download weights at request time. The `.joblib` files are pickle-based artifacts: only load artifacts from a trusted source.

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

The Python environment needs the same TabFM package/module version used to create `tabfm_final.joblib`, because the serialized estimator imports its original class when loaded. The model weights are contained in the artifact and are not downloaded or initialized from the official loader at runtime. **TabFM v1.0.0 weights have a separate non-commercial, non-production license**, distinct from the Apache-2.0 source license. Review the applicable license before use.

### TabPFN access

Install TabPFN via the requirements file. The package is needed to load and run the serialized estimator; its checkpoint weights are already in `tabpfn_final.joblib`, so the service does not download them at startup. Model-version licenses change; confirm that your intended use is allowed by the artifact's license.

The artifacts were serialized with scikit-learn 1.6.1. `python_backend\requirements.txt` pins that version because loading the saved preprocessing objects with other scikit-learn versions can break at inference time.

### Model artifacts

Keep the following files in the project-root `crop_yield_models` folder:

- `tabfm_final.joblib`
- `tabpfn_final.joblib`
- `numeric_imputer.joblib`
- `model_metadata.json`

The model files are large and excluded from Git. Put the matching artifacts in that folder on each machine that runs the backend. `CROP_YIELD_MODELS` can point to another directory if needed.

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

The service loads the data, saved imputer, and two pre-trained estimators at startup. No models are fit or evaluated at startup. Initial startup may take several minutes because the model artifacts are large. Watch the console for each model's readiness. Open `http://127.0.0.1:8000/health` to inspect service and model status, `http://127.0.0.1:8000/metrics` for metric availability, and `http://127.0.0.1:8000/docs` for the local API schema.

On CPU, a prediction using both models can also take several minutes. The XAMPP form allows up to six minutes for a submission and the PHP-to-API request allows up to five minutes; keep the Python backend running while the request completes.

In a second PowerShell window, from this project root, start the PHP site:

```powershell
php -S 127.0.0.1:8080
```

Open `http://127.0.0.1:8080`. For XAMPP, place this project folder under `C:\xampp\htdocs\` and use `http://localhost/Crop%20yield/` (or your chosen folder name); the Python API still runs separately on port 8000. PHP's `allow_url_fopen` must be enabled for the built-in API client.

## Endpoints

- `GET /health`: dataset and per-model readiness.
- `GET /metadata`: features, dataset-derived category lists, and evaluation protocol.
- `GET /metrics`: reports that held-out metrics are unavailable for the supplied full-data pre-trained artifacts.
- `POST /predict`: independently returns valid TabFM/TabPFN predictions, with an error status for a model that fails on that request.

Held-out metrics are intentionally not calculated: the supplied estimators were trained on all rows in the source dataset, so scoring them on a split of those same rows would be misleading. For an unbiased evaluation, provide separately held-out data and evaluate it independently. ROC AUC does not apply to these continuous regression predictions.

## Troubleshooting

- **API says the dataset is missing:** put the provided CSV in the project root or set `CROP_YIELD_DATA` to its full path.
- **A model artifact is unavailable:** confirm the corresponding `.joblib` file and `model_metadata.json` are in `crop_yield_models`; inspect the API terminal for the original exception.
- **TabFM cannot be loaded:** confirm the matching TabFM package/module is installed in the active environment so the saved estimator class can be imported.
- **Saved preprocessing fails:** install the pinned dependencies from `python_backend\requirements.txt`, especially scikit-learn 1.6.1.
- **PHP reports the service is not responding:** start the API on `127.0.0.1:8000`; check Windows Firewall and the PHP `allow_url_fopen` setting.
- **CUDA out-of-memory:** start with a CPU-only PyTorch environment or reduce other GPU workloads. The current implementation chooses CUDA when PyTorch reports it available; see the limitation below.
- **A model is unavailable:** the site can still show the other model's actual result.

## Current verification and limitations

The supplied notebook remains unchanged. Pre-trained estimators load independently, and startup logs each artifact's status. GPU-tagged tensors are remapped to the available device when loaded.
