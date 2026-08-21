# Forest-Loss Detection — Technical Results

## Project evolution

### Project 1: EuroSAT land-use baseline
- Model: pretrained ResNet18
- Task: 10-class land-use classification
- Dataset: 27,000 EuroSAT RGB images
- Training images: 18,900
- Validation images: 4,050
- Test images: 4,050
- Validation accuracy: 95.73%
- Test accuracy: 94.72%
- Forest-class accuracy: 92.31%

The classifier was later used as a heuristic change detector by comparing
forest classifications between 2018 and 2025 imagery. This demonstrated
the concept but was not trained directly for forest-loss segmentation.

### Project 2: Siamese U-Net
- Task: pixel-level forest-loss segmentation
- Inputs: paired 2018 and 2025 Sentinel-2 imagery
- Bands: B2, B3, B4, B8, B11, B12
- Training regions: 30
- Training patches: 10,303
- Validation regions: Sweden Central and Spain Galicia
- Final held-out tests: Australia NSW, Chile Biobio, Greece
- Patch size: 128×128
- Labels: Hansen Global Forest Change
- Loss: BCE + Tversky
- Tversky alpha: 0.4
- Tversky beta: 0.6
- Final validation-selected probability threshold: 0.15

## Engineering improvements
- Replaced land-cover-classification heuristics with end-to-end
  change segmentation.
- Expanded from RGB to six Sentinel-2 spectral bands.
- Used a shared Siamese encoder for two-date feature extraction.
- Computed absolute learned feature differences between dates.
- Added training-only per-band mean/std normalization.
- Added synchronized geometric augmentation.
- Added region-balanced and change-balanced sampling.
- Added learning-rate scheduling and early stopping.
- Tuned loss weighting and prediction threshold using held-out
  validation regions.
- Kept three geographic regions untouched for final testing.

## Final held-out results

Australia NSW:
- F1: 0.114
- Precision: 0.078
- Recall: 0.211
- Actual forest loss: 0.09%

Chile Biobio:
- F1: 0.538
- Precision: 0.408
- Recall: 0.787
- Actual forest loss: 12.53%

Greece:
- F1: 0.399
- Precision: 0.919
- Recall: 0.254
- Actual forest loss: 45.68%

Macro F1 across the three held-out regions: 0.350

## Generalization improvement

On the same unseen Greece stress test:

Initial Siamese U-Net:
- Precision: 0.914
- Recall: 0.012
- F1: 0.025

Final model:
- Precision: 0.919
- Recall: 0.254
- F1: 0.399

The largest remaining limitation is geographic calibration:
performance varies substantially between very-low-, medium-, and
high-loss environments.