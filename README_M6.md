# M6 Dual-Task Autoencoder

## Dataset

Dataset: PPG-DaLiA synchronized subject data.

Original source: UCI Machine Learning Repository, DOI 10.24432/C53890.

Download/extract the synchronized subject data before running the notebook.

Expected local structure:

data/PPG_FieldStudy/S1/S1.pkl
...
data/PPG_FieldStudy/S15/S15.pkl

The notebook does not modify raw data.

## Main protocol

- Train: S1-S9
- Validation: S10-S12
- Test: S13-S15
- BVP: 64 Hz
- Window: 8 s / 512 samples
- Stride: 4 s / 256 samples
- Main pipeline uses no digital band-pass filter
- Z-score uses Train-only global mean/std
- Main seeds: 42, 43, 44

## Expensive steps

Phase 7.3 trains the 15 main runs.

Phase 7.4 adds six gamma-sensitivity runs on Validation.

These are the expensive sections.

## Re-evaluation from checkpoints

Phase 8 and Phase 9 load the saved main checkpoints.
They do not retrain the models.

Test is not used for checkpoint selection.

## Outputs

- configs/split.json
- configs/label_map.json
- configs/scaler.json
- configs/experiment-config.json
- processed/m6-main-preprocessed.npz
- checkpoints/*.pt
- results/*.csv
- figures/*.png

All notebook figures are saved at 300 dpi.

## Final reproducibility check

Before submission:

1. restart the kernel;
2. Run All from top to bottom;
3. confirm all assertions pass;
4. save the executed notebook;
5. verify all files in the final audit table exist.
