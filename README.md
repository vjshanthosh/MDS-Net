Title: Multi Reference Face Verification

Overview: 
Given n images where the first n-1 images are the reference images (guaranteed to be the same) and the last image is the query image, determine whether the query image represents the same identity as the reference images. The models use a variant of Convolutional Neural Network. Models have been trained for n = 2, 3, 4, 5, and 6. All 5 models share the same training procedure and hyperparameters, while their architectures are similar but adapted to accept the corresponding number of images. An ablation study is performed to compare models for different n and determine which one provides the best tradeoff.

Dataset: 
The project used the CelebA dataset, which includes 202,599 face images across 10,177 identities. The images (img_align_celeba) could be downloaded from https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ and the identity list (identity_CelebA.txt) could be downloaded from https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ. img_align_celeba consists of 202,599 .jpg images with resolution of 178 x 218 pixels each. identity_CelebA.txt states which identity (integer) does each file path in the img_align_celeba correspond to. Place the img_align_celeba folder and identity_CelebA.txt inside the data folder so the get_data.ipynb file will work properly.

Data Pipeline & Preprocessing:
The 202,599 images are split into ~80% train, ~10% val, and ~10% test. These are identity disjoint splits with identities chosen at random for each split. The function get_data(n, size, split) loads size rows (or size-1 if size is odd) with n images per row in the file {split}_data/inputs_{n}.dat and {split}_data/targets_{n}.dat. Each image in the inputs file is of resolution 128 x 128 pixels (resized from 178 x 218) and has three channels (BGR). The targets file consists of 1 for matches and 0 for mismatches (# of matches = # of mismatches in each target file).

Training & Evaluation
The CNN variant model for each n value (2-6) are trained on 15 epochs. Some random minor transformations are applied to the train data on every epoch to prevent overfitting, but not to the val and test data. The epoch with the lowest binary cross entropy loss on the val split is selected for test set evaluation for each n value. The weights for the selected epochs are available in model_2/epoch_15.pth, model_3/epoch_15.pth, model_4/epoch_15.pth, model_5/epoch_14.pth, and model_6/epoch_14.pth. The weights for other epochs can also be accessed similarly, but they are not evaluated on the test set. 

Results:
n value | BCE loss on test set
------------------------------
2       | 0.3173
3       | 0.2584
4       | 0.2197
5       | 0.1993
6       | 0.2032
Model 5 (4 reference images, 1 query image) is considered as the model with the best balance.

System Requirements:
Python version: 3.9+ (tested on 3.12)
Core Libraries: PyTorch, Torchvision, OpenCV, and Scikit-Learn
MPS: works for both training and evaluation.
NVIDIA CUDA: works for both training and evaluation; change device = 'mps' to 'cuda' in train_models.ipynb and test_evaluate.ipynb and torch.mps.manual_seed to torch.cuda.manual_seed in train_models.ipynb.
CPU: works for evaluation only; change device = 'mps' to 'cpu' in test_evaluate.ipynb.

How to Run:
1. Download the dataset and identity list and place them in the data folder (as mentioned in dataset section).
2. Install the packages listed in requirements.txt.
3. Run get_data.ipynb to preprocess the data into the format required for training and evaluation.
4. Run train_models.ipynb to train the models for n = 2, 3, 4, 5, and 6.
5. Run test_evaluate.ipynb to evaluate the models on the test set; keep the device and hardware conditions consistent when calling inference_time function to get comparable results.

Repository Structure:
model_2, model_3, ... model_6: contains the weights for all epochs of the corresponding trained models.
get_data.ipynb: generates the preprocessed data as described in the Data Pipeline & Preprocessing section.
models.py: contains the neural network classes required to train, evaluate, or run each model from n=2 to n=6.
test_evaluate.ipynb: evaluates each model on the test split on metrics including BCE loss, accuracy, etc.
train_models.ipynb: to train the neural networks for each n value and includes validation results for each epoch.