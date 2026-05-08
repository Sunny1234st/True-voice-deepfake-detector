import os
import numpy as np
import librosa
import joblib
import scipy.stats
import scipy.fft
import warnings
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, learning_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Importing all required models
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==============================
# CONFIGURATION
# ==============================

TRAIN_DATA_PATH = "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-norm/for-norm"
TEST_FILE_PATH = "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-original/for-original/validation/fake/file758.mp3"

PIPELINE_FILE = "/kaggle/working/voice_auth_pipeline_best.pkl"
X_FILE = "/kaggle/working/X_data.npy"
y_FILE = "/kaggle/working/y_data.npy"

# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_lfcc(y, sr, n_lfcc=20):
    S = np.abs(librosa.stft(y))
    S_log = np.log(S + 1e-10)
    lfcc = scipy.fft.dct(S_log, axis=0, type=2, norm='ortho')[:n_lfcc]
    return lfcc

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=6)
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y, top_db=20)

        if len(y) < 0.5 * sr:
            return None

        S = np.abs(librosa.stft(y))

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        flux = librosa.onset.onset_strength(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        lfccs = extract_lfcc(y, sr, n_lfcc=20)
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)

        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[pitches > 0]

        pitch_mean = np.mean(pitch_vals) if len(pitch_vals) else 0
        pitch_std = np.std(pitch_vals) if len(pitch_vals) else 0
        pitch_range = (np.max(pitch_vals) - np.min(pitch_vals)) if len(pitch_vals) else 0

        mean_spectrum = np.mean(S, axis=1)
        skewness = scipy.stats.skew(mean_spectrum)
        kurtosis = scipy.stats.kurtosis(mean_spectrum)

        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel)

        features = np.hstack([
            np.mean(rolloff), np.std(rolloff),
            np.mean(centroid), np.std(centroid),
            np.mean(bandwidth), np.std(bandwidth),
            np.mean(flux), np.std(flux),
            np.mean(flatness), np.std(flatness),
            skewness, kurtosis,
            np.mean(zcr), np.std(zcr),
            np.mean(rms), np.std(rms),
            pitch_mean, pitch_std, pitch_range,
            
            np.mean(mfccs, axis=1), np.var(mfccs, axis=1),
            np.mean(mfcc_delta, axis=1), np.mean(mfcc_delta2, axis=1),
            np.mean(lfccs, axis=1), np.var(lfccs, axis=1),
            np.mean(chroma, axis=1), np.var(chroma, axis=1),
            np.mean(contrast, axis=1), np.var(contrast, axis=1),
            np.mean(mel_db, axis=1), np.var(mel_db, axis=1)
        ])

        return np.nan_to_num(features).astype(np.float32)

    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return None

# ==============================
# DATA LOADING
# ==============================

def load_limited_dataset(limit_per_class=500): 
    X, y = [], []
    classes = {'fake': 0, 'real': 1}

    for label_name, label_val in classes.items():
        folder = os.path.join(TRAIN_DATA_PATH, "training", label_name)
        
        if not os.path.exists(folder):
            print(f"[WARNING] Path missing: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.wav', '.mp3'))]
        np.random.shuffle(files)
        files = files[:limit_per_class]

        print(f"{label_name}: Extracting {len(files)} files...")

        for file in tqdm(files):
            feat = extract_features(os.path.join(folder, file))
            if feat is not None:
                X.append(feat)
                y.append(label_val)

    return np.array(X), np.array(y)

def get_feature_names():
    names = [
        "Rolloff_Mean", "Rolloff_Std", "Centroid_Mean", "Centroid_Std",
        "Bandwidth_Mean", "Bandwidth_Std", "Flux_Mean", "Flux_Std",
        "Flatness_Mean", "Flatness_Std", "Skewness", "Kurtosis",
        "ZCR_Mean", "ZCR_Std", "RMS_Mean", "RMS_Std",
        "Pitch_Mean", "Pitch_Std", "Pitch_Range"
    ]
    names += [f"MFCC_Mean_{i}" for i in range(1, 21)]
    names += [f"MFCC_Var_{i}" for i in range(1, 21)]
    names += [f"MFCC_Delta_Mean_{i}" for i in range(1, 21)]
    names += [f"MFCC_Delta2_Mean_{i}" for i in range(1, 21)]
    names += [f"LFCC_Mean_{i}" for i in range(1, 21)]
    names += [f"LFCC_Var_{i}" for i in range(1, 21)]
    names += [f"Chroma_Mean_{i}" for i in range(1, 13)]
    names += [f"Chroma_Var_{i}" for i in range(1, 13)]
    names += [f"Contrast_Mean_{i}" for i in range(1, 8)]
    names += [f"Contrast_Var_{i}" for i in range(1, 8)]
    names += [f"Mel_Mean_{i}" for i in range(1, 129)]
    names += [f"Mel_Var_{i}" for i in range(1, 129)]
    return names

# ==============================
# PIPELINE, EVALUATION & TRAINING
# ==============================

def train_and_evaluate():
    print("Checking for saved dataset...")
    
    if os.path.exists(X_FILE) and os.path.exists(y_FILE):
        print("Found saved data. Loading instantly...")
        X = np.load(X_FILE)
        y = np.load(y_FILE)
    else:
        print("No saved data found. Extracting features from audio files...")
        X, y = load_limited_dataset(limit_per_class=500)
        
        if len(X) == 0:
            print("[ERROR] No data loaded. Check Kaggle paths.")
            return
            
        np.save(X_FILE, X)
        np.save(y_FILE, y)
        print("Data saved successfully.")

    print(f"Total valid samples: {len(X)}")

    # Define the models to compare
    models = {
        "XGBoost": XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0, tree_method="hist", eval_metric="logloss", random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=50, random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model_name = ""
    best_accuracy = 0
    best_pipeline = None
    best_y_pred_cv = None

    print("\n" + "="*40)
    print("MODEL COMPARISON & ADVANCED METRICS")
    print("="*40)

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        # Cross Validation Predictions
        y_pred_cv = cross_val_predict(pipeline, X, y, cv=cv)
        
        # Calculate standard accuracy
        accuracy = np.mean(y_pred_cv == y)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_pipeline = pipeline
            best_y_pred_cv = y_pred_cv

        # Confusion Matrix Metrics (0 = Fake, 1 = Real)
        cm = confusion_matrix(y, y_pred_cv)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0          # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0          # False Negative Rate
        f1 = f1_score(y, y_pred_cv)                           # F1 Score

        print(f"Accuracy:    {accuracy * 100:.2f}%")
        print(f"F1 Score:    {f1:.4f}")
        print(f"Sensitivity: {sensitivity:.4f} (Ability to correctly identify Real voices)")
        print(f"Specificity: {specificity:.4f} (Ability to correctly identify Fake voices)")
        print(f"FPR:         {fpr:.4f} (Real humans falsely identified as Fake)")
        print(f"FNR:         {fnr:.4f} (Fake voices that slipped through as Real)")

    print("\n" + "="*40)
    print(f"WINNER: {best_model_name} with {best_accuracy * 100:.2f}% Accuracy")
    print("="*40)

    # =========================================================
    # ADVANCED DIAGNOSTICS (ROC, EER, THRESHOLD TUNING)
    # =========================================================
    print(f"\nGenerating Advanced Diagnostics for {best_model_name}...")
    
    # 1. Get probabilities instead of just 0 or 1 labels
    y_prob_cv = cross_val_predict(best_pipeline, X, y, cv=cv, method='predict_proba')[:, 1]
    
    # Calculate False Positive Rate, True Positive Rate, and Thresholds
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(y, y_prob_cv)
    roc_auc = auc(fpr_roc, tpr_roc)
    fnr_roc = 1 - tpr_roc
    
    # Calculate Equal Error Rate (EER)
    eer_index = np.nanargmin(np.absolute((fnr_roc - fpr_roc)))
    eer = fpr_roc[eer_index]
    eer_threshold = thresholds_roc[eer_index]
    
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Equal Error Rate (EER): {eer*100:.2f}% at Threshold {eer_threshold:.4f}")

    # 2. PLOT CONFUSION MATRIX
    disp = ConfusionMatrixDisplay.from_predictions(y, best_y_pred_cv, display_labels=["Fake", "Real"], cmap="Blues")
    plt.title(f"Confusion Matrix ({best_model_name})")
    plt.show()

    # 3. PLOT ROC CURVE
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_roc, tpr_roc, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.scatter(fpr_roc[eer_index], tpr_roc[eer_index], color='red', s=50, zorder=5, label=f'EER Operating Point')
    plt.title(f'ROC Curve ({best_model_name})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

    # 4. PLOT FNR vs FPR (THRESHOLD TUNING)
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds_roc, fpr_roc, label='FPR (False Alarms)', color='blue')
    plt.plot(thresholds_roc, fnr_roc, label='FNR (Missed Fakes)', color='coral')
    plt.axvline(x=eer_threshold, color='gray', linestyle='--', label=f'EER = {eer*100:.1f}%')
    plt.title('Threshold Tuning: FNR vs FPR')
    plt.xlabel('Decision Threshold (Probability of Real)')
    plt.ylabel('Error Rate')
    plt.xlim([0.0, 1.0])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # ---------------------------------------------------------
    # LEARNING CURVE GRAPH
    # ---------------------------------------------------------
    print(f"\nGenerating Learning Curve for {best_model_name}...")
    train_sizes, train_scores, test_scores = learning_curve(
        best_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label="Training Accuracy", marker='o', color='blue')
    plt.plot(train_sizes, test_mean, label="Cross-Validation Accuracy", marker='s', color='orange')
    plt.title(f"Learning Curve ({best_model_name})")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    # ---------------------------------------------------------
    # FINAL TRAINING
    # ---------------------------------------------------------
    print(f"\nTraining final {best_model_name} model on entire dataset...")
    best_pipeline.fit(X, y)
    joblib.dump(best_pipeline, PIPELINE_FILE)
    print(f"Pipeline saved as {PIPELINE_FILE}")

    # Feature importances (Only applies if winner is XGBoost or AdaBoost)
    if best_model_name in ["XGBoost", "AdaBoost"]:
        trained_model = best_pipeline.named_steps['classifier']
        importances = trained_model.feature_importances_
        feature_names = get_feature_names()

        important_features = sorted(zip(importances, feature_names), reverse=True)

        print("\nTop 10 Most Important Features:")
        for importance, name in important_features[:10]:
            print(f" - {name}: {importance:.4f}")

        top_10 = important_features[:10]
        plt.figure(figsize=(10, 6))
        plt.barh([x[1] for x in top_10][::-1], [x[0] for x in top_10][::-1], color='skyblue')
        plt.xlabel('Importance Score')
        plt.title(f'Top 10 Features ({best_model_name})')
        plt.show()


# ==============================
# PREDICTION
# ==============================

def predict_voice(file_path):
    if not os.path.exists(PIPELINE_FILE):
        print("Pipeline file not found. Training model first...\n")
        train_and_evaluate()

    if not os.path.exists(file_path):
        print(f"[ERROR] Test file not found: {file_path}")
        return

    pipeline = joblib.load(PIPELINE_FILE)

    feat = extract_features(file_path)
    if feat is None:
        print("Invalid audio file")
        return

    pred = pipeline.predict(feat.reshape(1, -1))[0]
    prob = pipeline.predict_proba(feat.reshape(1, -1))[0]

    label = "AI-GENERATED" if pred == 0 else "HUMAN"
    confidence = max(prob) * 100

    print("\n" + "="*30)
    print(f"TARGET: {os.path.basename(file_path)}")
    print(f"RESULT: {label}")
    print(f"CONFIDENCE: {confidence:.2f}%")
    print("="*30 + "\n")

# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    train_and_evaluate()
    predict_voice(TEST_FILE_PATH)
