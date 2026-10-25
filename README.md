# Deep Learning-Based Driver Drowsiness Detection

This capstone combines two lightweight image classifiers with a temporal decision layer:

1. An eye-state CNN detects closed eyes (MRL eye images).
2. A mouth-crop CNN detects yawning (weakly labelled crops from YawDD).
3. A rolling monitor reports `OK`, `CAUTION`, or `ALERT`.

Training can run on a Google Colab GPU; the live demo is intended for a local webcam.

## Requirements and setup

Use Python 3.12 and create/activate a virtual environment.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `.venv312` already exists, activate it instead of recreating it. Linux/macOS activation
is `source .venv312/bin/activate`. The same source code uses `pathlib` and relative paths on
Windows and Linux.

## Dataset downloads

- MRL Eye Dataset: <https://mrl.cs.vsb.cz/eyedataset.html>
- YawDD: IEEE DataPort, DOI <https://doi.org/10.21227/e1qm-hb90>

Download and extract each dataset locally. The MRL directory should contain the original
underscore-delimited PNG names. The YawDD directory should contain its driver videos
(Dash and Mirror collections); the collection, gender, and leading video identifier are
combined to keep reused numeric IDs in separate subjects.
Keep the downloaded datasets out of source control.

## Full pipeline

From the project root, replace the example paths with your extracted dataset locations.
Preparation writes CSV paths relative to each manifest file, so these manifests and their
generated crops can be moved together.

```powershell
python prepare_mrl.py --input-dir data\MRL
python prepare_yawdd.py --input-dir data\YawDD
```

Train both models (the default architecture is `improved`):

```powershell
python train.py --manifest data\mrl_manifest.csv --task eye --arch improved --epochs 30 --batch 32 --lr 0.001
python train.py --manifest data\yawdd_manifest.csv --task yawn --arch improved --epochs 30 --batch 32 --lr 0.001
```

Evaluate the test splits:

```powershell
python evaluate.py --manifest data\mrl_manifest.csv --task eye --arch improved
python evaluate.py --manifest data\yawdd_manifest.csv --task yawn --arch improved
```

To compare architectures, repeat training and evaluation with `--arch simple` and
`--arch mobilenet`. Models, histories, learning curves, confusion matrices, wrong-prediction
CSVs, and the appended `outputs/summary.csv` are written under `outputs/`.

Run the webcam demo after training:

```powershell
python demo.py --source 0 --eye-model outputs\eye_improved.keras --yawn-model outputs\yawn_improved.keras
```

Or exercise the camera/landmark/temporal pipeline before training either model:

```powershell
python demo.py --source 0 --heuristic
```

For a video file, pass its path to `--source`. Press `q` in the preview window to quit.
New ALERT transitions are appended to `outputs/alerts_log.csv`.

## Temporal rules

- ALERT: a continuous eye closure of at least 1.5 seconds, PERCLOS at least 0.40 in a
  rolling 10-second window (only after the window has at least 5 seconds of observations),
  or at least two sustained yawns in the last 60 seconds.
- CAUTION: exactly one recent sustained yawn, or ready-window PERCLOS above 0.20.
- Yawns count after at least 1 second of continuous yawning. Short eye closures remain blinks.

Run the temporal unit tests with `python test_temporal.py`.

## Limitations

- YawDD labels are weak labels: filename conditions and mouth-aspect-ratio thresholds do
  not provide precise frame-level ground truth. Talking, camera pose, and occlusion can
  cause noisy labels.
- MRL images are largely infrared; this creates an infrared-to-webcam domain gap that can
  reduce live-demo accuracy. Webcam-specific fine-tuning or calibration may be necessary.
- FaceMesh tracking, lighting, glasses, head pose, and camera placement affect crop quality.
- This is a university prototype, not a certified safety product or a substitute for
  driver attention, vehicle sensors, or professional safety validation.

capstone\
  data\
    MRL\
      ... MRL image files ...
    YawDD\
      ... Dash and Mirror video files ...