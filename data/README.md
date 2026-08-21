# Data

The raw datasets used in this project are not included in the repository because of their size.

## Project 1 — EuroSAT baseline

The first stage of the project used the EuroSAT land-use classification dataset.

- 27,000 RGB satellite image tiles
- 10 land-use classes
- 64 × 64 pixel images
- Used to train a pretrained ResNet18 classifier
- Train / validation / test split: 70% / 15% / 15%

The trained classifier was later applied to satellite imagery from different years as an initial heuristic approach to forest-loss detection.

## Project 2 — Sentinel-2 change detection

The final model uses paired Sentinel-2 imagery from 2018 and 2025.

Each image contains six spectral bands:

- B2 — Blue
- B3 — Green
- B4 — Red
- B8 — Near Infrared
- B11 — Short-Wave Infrared 1
- B12 — Short-Wave Infrared 2

Images were exported from Google Earth Engine and divided into 128 × 128 pixel patches.

## Labels

Pixel-level forest-loss labels were generated using the Hansen Global Forest Change dataset.

Pixels are represented as:

- `0` — no forest loss
- `1` — forest loss between 2018 and 2025

Hansen forest loss represents tree-cover loss and is not necessarily equivalent to permanent human-caused deforestation. Loss may also include events such as fire, harvesting, storms, or other disturbances.

## Geographic split

The final training dataset contains:

- 30 training regions
- 10,303 training patches
- 2 validation regions
- 3 held-out test regions

### Validation regions

- Sweden — Central
- Spain — Galicia

### Final test regions

- Australia — New South Wales
- Chile — Biobio
- Greece

The test regions were kept separate from model training and validation.

## Preprocessing

Sentinel-2 reflectance values were:

1. clipped to the range 0–10,000
2. divided by 10,000
3. standardized using per-band mean and standard deviation calculated from the training set only

Training augmentation included synchronized:

- 90° rotations
- horizontal flips
- vertical flips

The same transformation was applied to both dates and the target mask.

## Sampling

The choices made as seen above reduced two sources of imbalance:

1. geographic imbalance between regions with different numbers of patches
2. class imbalance between low-, medium-, and high-loss patches

## Data quality notes

Two expansion regions were excluded from final modeling:

- Brazil — Para: 2018 imagery required a different Sentinel-2 product, creating an inconsistent 2018/2025 pair
- Malaysia — Sarawak: only three usable patches remained after filtering invalid imagery

Some other regions contained small amounts of missing Sentinel-2 data. Patches with no more than 1% invalid pixels were retained and missing values were filled using the median value of the valid pixels in that patch.

## Reproducing the dataset

The repository does not currently include the full raw Sentinel-2 exports or generated NumPy patch files because they are too large for GitHub.

The notebooks document the processing workflow used to transform satellite imagery and Hansen masks into model-ready patches.
