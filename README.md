True Voice: Deepfake Audio Detection

Description
This project is an advanced machine learning pipeline designed to detect AI-generated synthetic audio (deepfakes) and distinguish it from organic human speech. Using digital signal processing (DSP) techniques, the system extracts high-frequency acoustic features—including Mel-Frequency Cepstral Coefficients (MFCCs), Linear Frequency Cepstral Coefficients (LFCCs), and Log-Mel Spectrograms—to capture the microscopic anomalies left behind by neural text-to-speech generators. 

The core classification is powered by an optimized Extreme Gradient Boosting (XGBoost) model. To ensure high security for voice-based biometric systems, the pipeline includes advanced diagnostics like Stratified 5-Fold Cross-Validation, ROC-AUC scoring, and Equal Error Rate (EER) threshold tuning to carefully balance false positives and false negatives.

---

 How to Run the Project

You can run this project either directly in a Kaggle Notebook (recommended) or locally on your own machine.

Option 1: Running on Kaggle (Easiest Method)
Since the code was originally built on Kaggle, the file paths are already set up for this environment..
1. Create a new Notebook on [Kaggle](https://www.kaggle.com/).
2. Click on Add Data in the right-hand menu.
3. Add the required dataset using this direct link: [The Fake or Real Dataset (FoR)](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset).
4. Upload a sample audio file to test, or use one from the dataset.
5. Paste the Python code into a cell.
6. Make sure your `TRAIN_DATA_PATH` and `TEST_FILE_PATH` match the Kaggle input directories, then click Run All.

 Option 2: Running Locally (On your own IDE)
If you want to run this on VS Code, PyCharm, or Jupyter Notebook on your own machine:

1. Clone the repository:
```bash
git clone [https://github.com/Sunny1234st/True-voice-deepfake-detector.git]
cd True-voice-deepfake-detector

```
2. Install the required dependencies:
Ensure you have Python installed, then run:
```bash
pip install numpy librosa joblib
 scipy matplotlib tqdm scikit-learn
 xgboost

```
3. Download the Dataset Manually:
  Download the dataset here: The Fake or Real Dataset: https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset?hl=en-IN
  Extract the .zip file on your computer.

4. Update the File Paths:
Open the Python script in your IDE and update the configuration section to point to the folders where you extracted the data:
```python

 Change these to your local computer paths
TRAIN_DATA_PATH = "C:/path/to/your/extracted_data/for-norm/training
TEST_FILE_PATH = "C:/path/to/your/sample_audio.mp3"


5. Run the code:
```bash
python your_script_name.py


(Note: The first time you run it, the code will take a few minutes to extract the mathematical features from the audio and save them as .npy files. Subsequent runs will load instantly!)
