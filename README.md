# CP467GroupProject
Group Project repository for CP467A, taught by prof. Zia Ud Din, Group members: David Hu, Jacob Jung, Roshni Mahindru, Paarth Bagga, Linda Basha
# Traffic Sign Recognition: Classical vs Deep Learning

This project compares a classical computer vision approach with a deep learning approach for traffic sign recognition using the German Traffic Sign Recognition Benchmark (GTSRB).

The goal is to evaluate not only classification accuracy, but also how robust each approach is under degraded image conditions such as reduced brightness, added noise, and motion blur.

## Approaches

### HOG + SVM
The classical pipeline resizes and converts images to grayscale, extracts Histogram of Oriented Gradients (HOG) features, and uses a Support Vector Machine (SVM) for classification.

### Convolutional Neural Network
The deep learning pipeline uses a CNN implemented in PyTorch to automatically learn visual features and classify traffic signs.

## Evaluation

Both models are evaluated using:

- Classification accuracy
- Macro F1-score
- Confusion matrices
- Training time
- Average inference time per image

Testing is performed on both the original GTSRB test set and degraded versions containing:

- Reduced brightness
- Added noise
- Motion blur

Additional preprocessing methods such as histogram equalization and Gaussian filtering may also be evaluated.

## Dataset

German Traffic Sign Recognition Benchmark (GTSRB)

- 43 traffic sign classes
- 39,209 training images
- 12,630 test images

Dataset source:  
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

The original/raw dataset is intentionally versioned in this repository. The group
also maintains a Google Drive copy for Colab. Do not commit separate brightness,
noise, blur, equalized, or filtered datasets: generate those images in memory with
the shared utilities.

**Upload status checked on November 22, 2026:** the GitHub/local copy is incomplete.
The existing files have been preserved exactly:

```text
Dataset/raw/
|-- Meta.csv                 # 43 class metadata rows
|-- Meta/
|   |-- 0.png ... 42.png     # 43 sign icons
|   |-- .~lock.ClassesInformation.ods#
|   `-- .~lock.ClassesInformationStrong.ods#
`-- Test/
    `-- 00000.png ... 06479.png  # 6,480 test images
```

Missing: `Train/` (all 39,209 training images), `Train.csv`, `Test.csv`, and
6,150 test images (`Test/06480.png` through `Test/12629.png`, inclusive).
`Meta.csv` describes classes; it does **not** label the training/test images.
Finish uploading the original source files, or point Colab at a complete Drive
copy. Do not invent labels or bypass validation to use this partial upload.

The loader supports the Kaggle layout below (case-sensitive on Colab/Linux).
Extract the download first; point at this folder, not its parent or a ZIP file.

```text
raw/
|-- Train.csv                # Path and ClassId columns; 39,209 rows
|-- Test.csv                 # Path and ClassId columns; 12,630 rows
|-- Train/0/ ... Train/42/   # original training images
|-- Test/                    # original test images
|-- Meta.csv                 # optional for model loading
`-- Meta/                    # optional sign icons
```

CSV image paths are relative to `raw/`, such as `Train/0/00000_00000_00000.png`
or `Test/00000.png`; labels must be integer IDs 0 through 42. This setup does not
translate other GTSRB distributions (for example, original PPM archives) automatically.

## Technologies

- Python
- OpenCV
- scikit-image
- scikit-learn
- PyTorch
- torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Google Colab

## Team

- Jacob Jung
- Linda Basha
- Paarth Bagga
- David Hu
- Roshni Mahindru

## Project Goal

The final comparison will examine the trade-off between the higher expected accuracy and robustness of CNNs and the lower computational cost and greater simplicity of a HOG + SVM pipeline.

## Repository Structure

```text
CP467GroupProject/
|-- Dataset/raw/                 # original GTSRB files; currently incomplete
|-- notebooks/
|   |-- 00_data_setup.ipynb      # environment, validation, summary, sample grid
|   |-- 01_hog_svm.ipynb         # HOG + SVM starter with TODO sections
|   `-- 02_cnn.ipynb             # PyTorch starter with device check and TODOs
|-- src/
|   |-- __init__.py
|   |-- config.py               # paths, seeds, image size, shared conditions
|   |-- data_utils.py           # metadata, RGB loading, validation, shared split
|   `-- degradations.py         # degradation and optional preprocessing helpers
|-- results/
|   |-- plots/
|   |-- confusion_matrices/
|   `-- metrics.csv             # header only until actual experiments run
|-- models/                     # reasonably sized final model artifacts
|-- tests/test_setup.py         # utility checks using temporary test fixtures
|-- .gitignore
`-- requirements.txt
```

This is project setup only. Model extraction/architecture, training, tuning,
evaluation, and final figures remain for the model teams to implement.

## Getting Started in Google Colab

1. Start with [00_data_setup in Colab](https://colab.research.google.com/github/Davy-Chu/CP467GroupProject/blob/main/notebooks/00_data_setup.ipynb).
2. In the first configuration cell, keep `USE_DRIVE = True` and change
   **`DRIVE_DATA_ROOT`** to the shared dataset folder. This is the **one dataset
   path teammates may need to change in each notebook**. The supplied
   `/content/drive/MyDrive/CP467_Traffic_Signs/data/raw` is an example only.
   If the shared folder appears under **Shared with me**, first add a shortcut
   under **My Drive** and use the shortcut's path.
3. Run the mount/setup cells and authorize Google Drive access in Colab. The
   notebook finds an existing code checkout or clones this repository into
   `/content/CP467GroupProject` and makes `src/` importable automatically.
   The code checkout and Drive dataset are separate locations.
4. Run the dependency imports. If a library is missing, set
   `INSTALL_DEPENDENCIES = True` in the dependency cell and rerun it. Restart
   the runtime if installation requests it, set the flag back to `False`, and
   rerun from the top. A working Colab environment needs no reinstall.
5. Run dataset validation, inspect the counts, class distribution and sample
   grid, and compare the printed **split ID** with teammates. Missing files,
   labels, classes, or incomplete counts stop execution with diagnostics.
   The default checks every file path and decodes one image per class per split;
   use `check_images="all"` to decode every image for a complete corruption scan.
6. Choose your model notebook:
   - [01_hog_svm in Colab](https://colab.research.google.com/github/Davy-Chu/CP467GroupProject/blob/main/notebooks/01_hog_svm.ipynb):
     set the same Drive path, run setup/data loading, and implement the TODO sections.
   - [02_cnn in Colab](https://colab.research.google.com/github/Davy-Chu/CP467GroupProject/blob/main/notebooks/02_cnn.ipynb):
     choose **Runtime → Change runtime type → GPU**, set the same Drive path,
     run through the device/data checks, and implement the TODO sections.

No external API key, Kaggle key, or Google credentials file is needed for training
with the supplied files. Drive authorization happens interactively in Colab;
never save tokens or passwords in notebooks.

**Code already on Drive:** after mounting Drive, change to the folder containing
`src/` and `requirements.txt` before running the project-code setup cell:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/CP467GroupProject
```

That example is the **code checkout** path; adjust it only if you intentionally
keep the code on Drive. The default automatic clone needs no code-path change.
To use an existing Colab clone instead, use `%cd /content/CP467GroupProject`.
Cloning happens only when absent: to update an existing clone, save your edits
first, then run `git pull --ff-only` from its root.

### Local fallback

From a cloned project root, create a virtual environment and install dependencies:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell (use instead of source):
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open the notebooks using your notebook editor with this environment selected
(install `ipykernel` if the editor requires it). They find the project root from
the root or `notebooks/` working directory. Local runs default to
`USE_DRIVE = False`, selecting `Dataset/raw/`. Complete the raw upload first.
No machine-specific absolute path is embedded in the project.

## Shared Experiment Rules

- `src/config.py` defines 43 classes, 32×32 RGB input, seed 42, and a 20%
  validation fraction. `seed_everything()` seeds Python, NumPy, and PyTorch.
- Both notebooks call `make_train_validation_split(training_metadata)`.
  It sorts by relative path and makes the same stratified 80/20 split with seed
  42, independent of CSV row order or the Drive/local absolute root. With the
  complete dataset this yields 31,367 training and 7,842 validation images.
  Record `SPLIT_ID` (a fingerprint of the actual rows/labels) with results;
  compare it across teams before training. Keep library versions consistent
  if split IDs differ. Do not create another split inside either pipeline.
- Keep the official 12,630-image test set untouched until final evaluation.
  Model choices, preprocessing choices, and fitted scalers use training and
  validation only. No test-set tuning, including on degraded test images.
- `load_image` returns resized **RGB uint8**. Use `apply_condition(image,
  condition, image_id=row.Path)` on that common image, then optional shared
  preprocessing, then model-specific grayscale/HOG or tensor normalization.
  This fixes transform order and input size across both pipelines.
- Conditions are `clean`, `brightness` (factor 0.5), `noise` (Gaussian standard
  deviation 0.05 on [0,1]), and `motion_blur` (9-pixel horizontal kernel).
  These are shared starting severities, not tuned model hyperparameters.
  Change `DEGRADATION_PARAMS` together if the experiment plan changes.
  Noise uses seed 42 plus a stable hash of the CSV-relative image path, so
  shuffling or batching does not change the noisy image. Do not use an absolute
  path or batch index as `image_id`.
- All transforms preserve dtype, shape, and valid intensity range; they accept
  uint8 [0,255] or floating [0,1], in grayscale or RGB. Optional
  `equalize_histogram` and `gaussian_filter` are not automatic. Float histogram
  equalization uses 8-bit quantization. Record preprocessing and parameters
  for each experiment so results with/without preprocessing remain distinct.
- GPU determinism can vary across versions/devices. Future DataLoaders must
  seed their generators and any random workers; the starter marks this TODO.

## Results and Model Storage

`results/metrics.csv` starts with a header only. Append measurements after real
experiments, with one row per model/condition/preprocessing combination:

| Column | Definition |
| --- | --- |
| `model` | `hog_svm` or `cnn` |
| `condition` | `clean`, `brightness`, `noise`, or `motion_blur` |
| `preprocessing` | `none` or the optional preprocessing name and parameters |
| `accuracy` | Fraction correct, in [0,1] |
| `macro_f1` | Macro F1 over labels 0–42, `zero_division=0`, in [0,1] |
| `training_time_seconds` | Elapsed training seconds; same trained model value across test conditions |
| `inference_time_ms` | Mean elapsed milliseconds per test image |
| `seed` | Shared seed, initially 42 |
| `split_id` | Printed shared split fingerprint |

Use explicit labels 0–42 for both macro F1 and confusion matrices. Save plots in
`results/plots/` and matrices/figures in `results/confusion_matrices/`. In the
experiment notebook, record library versions, hardware, model settings,
degradation parameters, batch size, timing boundaries, and whether preprocessing
is included. Agree on timing boundaries and hardware across both teams; warm up
inference and synchronize CUDA before/after GPU timing.

Reasonably sized final artifacts can be committed as `models/hog_svm.pkl` and
`models/cnn.pt`; do not create placeholders with those extensions. Put temporary
checkpoints under `models/checkpoints/` (ignored), and keep unnecessarily large
artifacts on Drive with their location documented. The `models/`, `Dataset/`,
`notebooks/`, `src/`, and `results/` directories themselves remain trackable.

## Development Workflow

Work/run experiments in Colab → save notebook → commit updated `.ipynb`, shared
code, and real results to GitHub. In Colab, use **File → Save a copy in GitHub**
or download the notebook and commit it from your local checkout. Prefer a feature
branch and a pull request for review; edits to cloned `src/` files also need to
be saved and committed. The Colab `/content` filesystem is temporary, so save
important code and artifacts before ending a session. Avoid committing noisy
outputs, secrets, or generated dataset copies.

Run the shared utility checks locally from the root after installing requirements:

```bash
python -m unittest discover -s tests -v
```

These use small, temporary synthetic test fixtures only; they never replace or
write to `Dataset/raw/`. Real dataset readiness is checked in `00_data_setup`.

## Team Workstreams

- HOG + SVM
- CNN
- Shared preprocessing and degradation
- Evaluation and results
- Report and documentation

Coordinate shared split, transformation, and metric changes before running
experiments. Individual responsibilities remain as agreed by the group.
