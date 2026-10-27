Title: Multi-branch Distance Sorting Network (MDS-Net) for Multi Reference Face Verification

Overview: 
The objective of MDS-Net: given n images where the first n-1 images are the reference images (guaranteed to be the same identity) and the last image is the query image, determine whether the query image represents the same identity as the reference images. Models have been trained for n = 2, 3, 4, 5, and 6. The architecture involves converting each image to an embedding using convolutional layers, computing the distance between each reference embedding and the query embedding, sort the distances by magnitude and concatenate, and pass the concatenated vector into fully connected layers. The MDS-Net achieves statistically significant improvement over conventional baselines. 

Dataset: 
The project used the CelebA dataset, which includes 202,599 face images across 10,177 identities. The images (img_align_celeba) could be downloaded from https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ and the identity list (identity_CelebA.txt) could be downloaded from https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ. img_align_celeba consists of 202,599 .jpg images with resolution of 178 x 218 pixels each. identity_CelebA.txt states which identity (integer) does each file path in the img_align_celeba correspond to. Place the img_align_celeba folder and identity_CelebA.txt inside the data folder so the get_data.ipynb file will work properly.

Data Pipeline & Preprocessing:
The 202,599 images are split into ~80% train, ~10% val, and ~10% test. These are identity disjoint splits with identities chosen at random for each split. The function get_data(n, size, split) loads size rows (or size-1 if size is odd) with n images per row in the file {split}_data/inputs_{n}.dat and {split}_data/targets_{n}.dat. Each image in the inputs file is of resolution 128 x 128 pixels (resized from 178 x 218) and has three channels (BGR). The targets file consists of 1 for matches and 0 for mismatches (# of matches = # of mismatches in each target file).

Training & Evaluation
The MDS-Net model for each n value (2-6) are trained on 15 epochs. Some random minor transformations are applied to the train data on every epoch to prevent overfitting, but not to the val and test data. The epoch with the lowest binary cross entropy loss on the val split is selected for test set evaluation for each n value. The weights for the selected epochs are available in model_2/epoch_15.pth, model_3/epoch_15.pth, model_4/epoch_15.pth, model_5/epoch_14.pth, and model_6/epoch_14.pth. The weights for other epochs can also be accessed similarly, but they are not evaluated on the test set. 

Results:
The table below compares the Binary Cross Entropy Loss against Best Cosine Embedding (BCS) baseline and Average Siamese Probability (ASP) baseline. BCS applies a logistic regression on the highest cosine similarity between a reference image and the query image. ASP gives the average of the probability for the n=2 model.
n value | MDS-Net BCE loss | BCS BCE loss | ASP BCE loss
---------------------------------------------------------
2       | 0.3132           | 0.4326       | 0.3132
3       | 0.2533           | 0.3718       | 0.2782
4       | 0.2249           | 0.3719       | 0.2723
5       | 0.2139           | 0.3667       | 0.2669
6       | 0.2044           | 0.3700       | 0.2667

System Requirements:
Python version: 3.9+ (tested on 3.12)
Core Libraries: PyTorch, Torchvision, OpenCV, and Scikit-Learn
MPS: works for both training and evaluation.
NVIDIA CUDA: works for both training and evaluation; change device = 'mps' to 'cuda' in models.py, train_models.ipynb, baseline.ipynb, test_evaluate.ipynb, and bootstrap.ipynb and torch.mps.manual_seed to torch.cuda.manual_seed in train_models.ipynb.
CPU: works for evaluation only; change device = 'mps' to 'cpu' in test_evaluate.ipynb and bootstrap.ipynb.

How to Run:
1. Download the dataset and identity list and place them in the data folder (as mentioned in dataset section).
2. Install the packages listed in requirements.txt.
3. Run get_data.ipynb to preprocess the data into the format required for training and evaluation.
4. Run train_models.ipynb to train the models for n = 2, 3, 4, 5, and 6 and the models used for additional experiments.
5. Run baseline.ipynb to train the baseline model.
6. Run test_evaluate.ipynb to evaluate the models on the test set.
7. Run bootstrap.ipynb to generate the confidence intervals.

Repository Structure:
model_2, model_3, ... model_6: contains the weights for all epochs of the corresponding trained models.
get_data.ipynb: generates the preprocessed data as described in the Data Pipeline & Preprocessing section.
models.py: contains the neural network classes required to train, evaluate, or run each model from n=2 to n=6, a baseline model, and the models used for additional experiments.
test_evaluate.ipynb: evaluates each model on the test split on metrics including BCE loss, accuracy, etc.
train_models.ipynb: to train the models for each n value and the models used for additional experiments.
bootstrap.ipynb: helps construct the 95% confidence interval used for comparison against baseline and additional experiments.