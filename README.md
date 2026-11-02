# Multi-branch Distance Sorting Network (MDS-Net) for Multi-Reference Face Verification

## Overview

The objective of MDS-Net is to determine whether a query image represents the same identity as a set of reference images.

Given *n* images, the first *n − 1* images are reference images (guaranteed to belong to the same identity), and the final image is the query image. The model determines whether the query image belongs to the same identity as the reference images.

Models have been trained for *n* = 2, 3, 4, 5, and 6.

The architecture converts each image into an embedding using convolutional layers, computes the distance between each reference embedding and the query embedding, sorts the distances by magnitude, and concatenates them. The resulting vector is then passed through fully connected layers to make the final prediction.

MDS-Net achieves statistically significant improvements over conventional baselines.

## Dataset

The project uses the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), which contains 202,599 face images across 10,177 identities.

The dataset images (`img_align_celeba`) and identity list (`identity_CelebA.txt`) can be downloaded from the following links:

- **Images:** [Download img_align_celeba](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ)
- **Identity list:** [Download identity_CelebA.txt](https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ)

The `img_align_celeba` folder contains 202,599 `.jpg` images, each with a resolution of 178 × 218 pixels. The `identity_CelebA.txt` file specifies the identity (integer) corresponding to each image file in `img_align_celeba`.

Place the `img_align_celeba` folder and `identity_CelebA.txt` inside the `data` folder for `get_data.ipynb` to work properly.

## Data Pipeline & Preprocessing

The 202,599 images are divided into approximately:

- 80% training data
- 10% validation data
- 10% test data

The splits are identity-disjoint, with identities randomly assigned to each split.

The function `get_data(n, size, split)` loads `size` rows (or `size - 1` if `size` is odd), with *n* images per row, from the following files:

- `{split}_data/inputs_{n}.dat`
- `{split}_data/targets_{n}.dat`

Each image in the inputs file has a resolution of 128 × 128 pixels (resized from 178 × 218) and three color channels (BGR).

The targets file contains binary labels:

- `1` for matches
- `0` for mismatches

Each target file contains an equal number of matches and mismatches.

## Training & Evaluation

The MDS-Net models for each *n* value (2–6) are trained for 15 epochs.

Random minor transformations are applied to the training data during each epoch to prevent overfitting. These transformations are not applied to the validation or test data.

For each *n* value, the epoch with the lowest binary cross-entropy (BCE) loss on the validation split is selected for evaluation on the test set.

The selected model weights are available at:

| n | Selected model weights |
|---|---|
| 2 | `model_2/epoch_14.pth` |
| 3 | `model_3/epoch_15.pth` |
| 4 | `model_4/epoch_15.pth` |
| 5 | `model_5/epoch_14.pth` |
| 6 | `model_6/epoch_15.pth` |

Weights from other epochs are also available in their respective model directories, but they are not evaluated on the test set.

## Results

The following table compares the Binary Cross-Entropy (BCE) loss of MDS-Net against two baselines:

- **Best Cosine Similarity (BCS):** Applies logistic regression to the highest cosine similarity between a reference image and the query image.
- **Average Siamese Probability (ASP):** Computes the average probability predicted by the *n* = 2 model.

| n | MDS-Net BCE Loss | BCS BCE Loss | ASP BCE Loss |
|---|---:|---:|---:|
| 2 | 0.3132 | 0.4326 | 0.3132 |
| 3 | 0.2533 | 0.3718 | 0.2782 |
| 4 | 0.2249 | 0.3719 | 0.2723 |
| 5 | 0.2139 | 0.3667 | 0.2669 |
| 6 | 0.2044 | 0.3700 | 0.2667 |

## System Requirements

- **Python:** 3.9+ (tested with 3.12)
- **Core libraries:** PyTorch, Torchvision, OpenCV, and Scikit-Learn

### Supported Devices

**Apple MPS**

- Supported for both training and evaluation.

**NVIDIA CUDA**

- Supported for both training and evaluation.
- In the following files, change `device = 'mps'` to `device = 'cuda'`:
  - `models.py`
  - `train_models.ipynb`
  - `baseline.ipynb`
  - `test_evaluate.ipynb`
  - `bootstrap.ipynb`
- In `train_models.ipynb`, change `torch.mps.manual_seed` to `torch.cuda.manual_seed`.

**CPU**

- Supported for evaluation only.
- In the following files, change `device = 'mps'` to `device = 'cpu'`:
  - `test_evaluate.ipynb`
  - `bootstrap.ipynb`

## How to Run

1. Download the dataset images and identity list, and place them inside the `data` folder as described in the [Dataset](#dataset) section.
2. Install the required packages listed in `requirements.txt`.
3. Run `get_data.ipynb` to preprocess the data into the format required for training and evaluation.
4. Run `train_models.ipynb` to train the models for *n* = 2, 3, 4, 5, and 6, as well as the models used for additional experiments.
5. Run `baseline.ipynb` to train the baseline model.
6. Run `test_evaluate.ipynb` to evaluate the models on the test set.
7. Run `bootstrap.ipynb` to generate the confidence intervals.

## Repository Structure

| File/Directory | Description |
|---|---|
| `model_2/`, `model_3/`, ..., `model_6/` | Contain the weights for all epochs of the corresponding trained models. |
| `get_data.ipynb` | Generates the preprocessed data described in the Data Pipeline & Preprocessing section. |
| `models.py` | Contains the neural network classes required to train, evaluate, or run each model for *n* = 2–6, a baseline model, and models used for additional experiments. |
| `test_evaluate.ipynb` | Evaluates each model on the test split using metrics including BCE loss, accuracy, and others. |
| `train_models.ipynb` | Trains the models for each *n* value and the models used for additional experiments. |
| `bootstrap.ipynb` | Constructs the 95% confidence intervals used for comparisons against baselines and additional experiments. |