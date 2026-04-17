# Imaged-Based-Reconstruction-model
This repository implements an image-based pipeline for vessel trajectory completion from AIS-derived trip data. It first preprocesses raw trips into a clean and temporally aligned representation, converts them into H3-aware RGB wave-map images, trains a U-Net model to inpaint missing trajectory regions, and finally reconstructs missing vessel positions back into geographic coordinates for evaluation on unseen holdout trips. The workflow is organized around a preprocessing notebook, a training script, an inference/reconstruction notebook, and a shared utility module that contains the common data transformation, encoding, quantization, training, and evaluation logic.

## Installation

Pre-requisites:

```sh
$ pip install -r requirements.txt
```

If you are running this through a notebook, you might need to restart the kernel to load changes.

Please find the end-to-end workflow in the preprocessing notebook for data preparation and wave-map generation, in the training script for model fitting and holdout evaluation, and in the inference notebook for trajectory reconstruction and DTW-based assessment.

## Framework

This framework enables (i) preprocessing raw AIS trips into fixed-size H3-colored wave maps, (ii) training an H3-aware inpainting network over masked trajectory images, and (iii) reconstructing missing trip segments on unseen holdout trajectories.

### (i) Preprocessing and wave-map construction

This module accepts raw trip records in CSV form and constructs a standardized representation suitable for image-based learning. More specifically:

* _Column normalization and parsing_: Raw input fields are renamed to a common schema with trip identifier, timestamp, longitude, latitude, and H3 cell columns. Timestamps are parsed robustly and trip durations are computed in order to filter out anomalous trajectories.
* _Trip cleaning_: Trips with durations outside a configurable tolerance window are excluded so that the downstream model is trained on representative vessel movements rather than outliers or incomplete sequences.
* _Temporal augmentation_: Each trip is resampled onto a fixed temporal grid, filling missing time bins by interpolation and recomputing H3 cells where necessary so that all trips share the same temporal support.
* _H3-aware color encoding_: The pipeline builds deterministic H3 color and position maps using a bit-packed float32 RGB encoding, allowing every H3 cell to be represented by a unique color while preserving a geographic relationship between nearby cells.
* _Wave-map export_: The augmented trips are transformed into fixed-size RGB wave maps stored as NumPy arrays, together with reusable JSON files that contain the H3-to-color and H3-to-position mappings required during training and inference.

### (ii) H3-aware inpainting model training

The training pipeline frames missing trajectory recovery as an image inpainting task. A deterministic percentage-based holdout split is created from the generated wave maps, and the training set is augmented with synthetic masks of different sizes and locations so that the model learns to reconstruct missing trajectory regions under diverse gap patterns. The model itself is a U-Net that receives a four-channel input composed of the masked RGB image and a binary mask channel. Training uses a reconstruction objective that emphasizes both the full image and the masked region, and predicted RGB values are quantized back to the nearest valid H3 colors through a KD-tree over the learned H3 color vocabulary. The script saves checkpoints, training-loss plots, the holdout manifest, and summary metrics such as masked RMSE and H3 classification accuracy on unseen holdout images.

### (iii) Inference and trajectory reconstruction

The inference workflow reconstructs missing trip segments at trajectory level rather than only at image level. For each holdout trip, the notebook can inject an artificial gap, fill small gaps by classical interpolation, and pass the remaining masked regions through the trained inpainting model after converting the trip into its wave-map representation. The inpainted RGB output is then translated back into valid H3 cells and geographic coordinates using the stored H3 color and position maps. This yields a reconstructed trajectory in longitude and latitude space, which is evaluated against ground truth using DTW over longitude, latitude, and combined two-dimensional geometry, along with H3 match rate and fill rate. The final outputs include reconstructed trip CSV files and interactive Folium maps for visual inspection.

## License

The contents of these repository are licensed under [GNU General Public License v3.0](https://github.com/M3-Archimedes/Imaged-Based-Reconstruction-model/blob/main/LICENSE).
