# Machine Learning-Based ECG Signal Classification for Arrhythmia Detection

Classifies ECG heartbeats from the MIT-BIH Arrhythmia Database into five
AAMI classes (N, S, V, F, Q) using a bandpass filter, feature extraction and a
Random Forest classifier. Includes a Streamlit web interface.

> Academic prototype - not a clinically validated diagnostic system.

## Dataset
Download from https://physionet.org/content/mitdb/1.0.0/ and extract the files
(`100.dat`, `100.hea`, `100.atr`, ...) anywhere inside the `data/` folder.

## Setup
```
pip install -r requirements.txt
```

## Run
```
python train.py                    # train + evaluate, saves model and reports
streamlit run app.py               # web interface
python predict.py --make_sample    # creates beats.csv
python predict.py --csv beats.csv  # command-line prediction
```
Quick test: `python train.py --max_records 8`

## Pipeline
1. Load records (MLII lead) and beat annotations (wfdb)
2. Bandpass filter 0.5-40 Hz (Butterworth)
3. Segment 200-sample window around each R peak
4. Extract features (statistics, shape, RR intervals)
5. Random Forest with class weighting
6. GroupShuffleSplit - train/test split by record (no leakage)
7. Classification report, accuracy, confusion matrix saved in `reports/`

## Classes
| Class | Description |
|---|---|
| N | Normal heartbeat group |
| S | Supraventricular ectopic beat |
| V | Ventricular ectopic beat |
| F | Fusion beat |
| Q | Other / unknown beat |

Paced records (102, 104, 107, 217) are excluded, following AAMI recommendation.
