# Deepfake Audio Detection for Voice Spoofing Attacks - Checkpoint #1

This project implements baseline deepfake-audio detection models using the ASVspoof 2019 dataset.

The program extracts log-mel spectrogram features from audio files and trains two models:

- A convolutional neural network (CNN)
- A bidirectional long short-term memory network (BiLSTM)

The models are evaluated using accuracy, balanced accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and Equal Error Rate (EER).

## Recommended Environment

The notebook is designed to run in Google Colab. 

## Files

The main and only project file is:

```text
CS228_checkpoint1.ipynb
```

## Requirements
As a new user, you do not need many prerequisites.

The notebook downloads the dataset during execution. The dataset does not need to be committed to the repository.
The notebook installs the Kaggle API automatically. Google Colab already provides most of the other required packages. 

## Project Setup Instructions

### 1. Open the Notebook

Open Google Colab:

https://colab.research.google.com/

Upload and/or open `CS228_checkpoint1.ipynb`.

### 2. Create Kaggle Credentials

The notebook downloads the ASVspoof dataset through Kaggle.

1. Sign into https://www.kaggle.com/ 
2. Go to account settings.
3. Locate the API section.
4. Create an API token or retrieve your Kaggle username and API key.

### 3. Add Kaggle Credentials to Colab

In Google Colab:

1. Open the Secrets tab using the key icon on the left side of the notebook.
2. Create a secret named:

```text
KAGGLE_USERNAME
```

3. Create another secret named:

```text
KAGGLE_KEY
```

4. Enable notebook access for both secrets.

The names must match exactly because the notebook reads the credentials using:

```python
userdata.get("KAGGLE_USERNAME")
userdata.get("KAGGLE_KEY")
```

Do not place Kaggle credentials directly inside the notebook or commit them to the repository.

## Running the Notebook

Run the notebook cells in order.

### Cell 0: Environment and Dataset Setup

This cell:

- Imports the required libraries
- Sets the random seed to 42
- Installs the Kaggle API
- Downloads the ASVspoof 2019 dataset
- Extracts the dataset into the Colab `/content` directory

### Cell 2: Feature Extraction

This cell loads the audio files and converts them into log-mel spectrograms.

The audio is:

- Loaded at 16 kHz
- Limited to three seconds
- Converted into a mel-spectrogram
- Converted to decibel values
- Padded or truncated to fixed dimensions
- Normalized to approximately the range [0, 1]

The CNN receives features shaped like:

```text
150 x 150 x 3
```

The BiLSTM receives features shaped like:

```text
150 x 128
```

### Cell 3: Model Definitions

This cell defines the CNN and BiLSTM architectures.

### Cell 4: Evaluation Functions

This cell defines the EER calculation and plotting functions.

### Cell 5: Model Initialization

This cell checks that both models can be created successfully and displays their architectures.

### Cell 6: Data Loading, EDA, and Training

This cell:

- Reads labels from the ASVspoof protocol file
- Loads up to 1,000 audio files
- Extracts CNN and BiLSTM features
- Displays the class distribution
- Displays example spectrograms
- Displays feature-statistics plots
- Creates a stratified 80/20 training-validation split
- Trains the CNN for five epochs
- Trains the BiLSTM for five epochs

### Cell 7: Evaluation and Error Analysis

This cell:

- Generates model predictions
- Calculates accuracy and balanced accuracy
- Calculates precision, recall, and F1-score
- Prints confusion-matrix counts
- Calculates ROC-AUC and EER
- Displays confusion matrices and ROC curves
- Lists examples of misclassified audio files

## Dataset Labels

The notebook converts the ASVspoof labels into binary values:

```text
0 = Spoof/Fake
1 = Bonafide/Real
```

## Expected Dataset Summary

The current experiment uses 1,000 files from the ASVspoof 2019 Logical Access training directory:

| Class | Number of Samples |
|---|---:|
| Spoof/Fake | 902 |
| Bonafide/Real | 98 |
| Total | 1,000 |

The data is split into:

| Split | Number of Samples |
|---|---:|
| Training | 800 |
| Validation | 200 |

## Important Interpretation of Results

The dataset is strongly imbalanced toward spoofed samples. As a result, accuracy alone can be misleading. A model that predicts every validation sample as spoofed can achieve 90% accuracy because 90% of the validation data is spoofed.

For this reason, balanced accuracy, confusion matrices, ROC-AUC, and EER should be considered when evaluating the models.

## Expected Output

After successfully running the notebook, it should produce:

- A class-distribution plot
- Example real and spoof spectrograms
- Feature-statistics plots
- Training-loss and training-accuracy plots
- CNN and BiLSTM confusion matrices
- CNN and BiLSTM ROC curves
- Accuracy, balanced accuracy, precision, recall, and F1-score
- ROC-AUC and EER values
- A list of misclassified audio files

## Potential Error(s)
### Out-of-Memory Error

If Colab runs out of memory, reduce the number of files processed by changing:

```python
selected_files = all_files[:1000]
```

For example:

```python
selected_files = all_files[:500]
```
