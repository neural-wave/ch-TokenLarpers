# Voltera: PV detection from electricity consumption profiles

This is our submission for the AIL NeuralWave challenge. For every customer in a 15-minute consumption CSV, `inference.py` predicts:

- whether a photovoltaic (PV) system is installed;
- if so, the commissioning date and nominal capacity (kWp) of the initial installation.

It also writes the optional PV-extension predictions.

The pipeline uses only the consumption time series and calendar features derived from its timestamps. It uses no weather, irradiation or location data, no LLM, and no network access.

- Method, evaluation and results: [`voltera-report.pdf`](voltera-report.pdf).
- Team: Alessandro Tognoni, Jeffre Daniel Martinez Marroquin, Tarun Deepak.

## Dependencies

- **Python:** 3.12 and 3.13 are both tested. 3.11 is covered by the pinned versions but untested. 3.14 is not supported.
- **Packages:** `requirements.txt` pins every package and is compiled from `requirements.in`. The direct dependencies are numpy, pandas, scipy, scikit-learn, pyarrow, ruptures, tzdata, CatBoost and PyTorch.
- **Hardware:** no GPU is needed; inference runs on the CPU. On Apple silicon, PyTorch uses the MPS backend instead; in our tests its output files were byte-identical to the CPU run.

## Environment setup

From this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This is the only step that needs internet access. On Linux x86_64, pip also installs PyTorch's CUDA libraries (several GB), which inference does not use.

## Run

From this folder:

```bash
python inference.py path/to/test_set.csv
```

- **Input:** one time-series CSV with the columns `customer_id`, `energy_consumption_kwh`, `timestamp_local` and `timestamp_epoch`, separated by `;` or `,`. Daylight-saving days with 92 or 100 intervals are handled.
- **Model:** loaded automatically from `artifacts/`. It was trained on the 40 labelled development customers.
- **Output:** two files in the current working directory, UTF-8 with `;` as separator and `.` as decimal mark.
  - `predictions.csv`: `customer_id;pv_installed;pv_installation_date;pv_capacity_kwp`.
    - One row per customer ID in the input. IDs stay strings, so leading zeros are kept.
    - `pv_installed` is `TRUE`/`FALSE`; dates are `YYYY-MM-DD`.
    - Date and capacity are empty when `pv_installed` is `FALSE`.
  - `pv_extensions_predictions.csv`: `customer_id;extension_date;pv_capacity_after_extension_kwp`. One row per predicted extension; the file may hold only its header.
- **Runtime:** about 10 seconds for 20 customers on a laptop. No manual step is needed, and repeated runs give byte-identical files.

The script also works from another folder (`python /path/to/submission/inference.py test.csv`). The outputs then go to that folder.

### Check the installation without real data

```bash
python -m pvdetect.synthetic /tmp/pv_demo --n 20 --seed 2
python inference.py /tmp/pv_demo/energy.csv
```

The first command writes 20 fabricated customers. The second must write both output files.

## Contents

|Path|Content|
|---|---|
|`inference.py`|Entry point|
|`pvdetect/`|Data loading and validation, preprocessing, features, models, the training procedure, and the synthetic data generator used in the check above|
|`artifacts/`|The trained model: `model.json` and the files it references (`catboost_clf.cbm`, `catboost_ext.cbm`, `fraunhofer.pt`)|
|`requirements.txt`, `requirements.in`|Pinned dependencies and the list they are compiled from|
|`voltera-report.pdf`|Final report|
