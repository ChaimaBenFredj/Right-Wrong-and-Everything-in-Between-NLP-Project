# This file Uses BERT embeddings + SVM classifier to predict response strategies

import os
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#-------------------- Config ----------------------------------

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
SEED = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#------------------- BERT embedding extractor ---------------------------

def get_bert_embeddings(texts, tokenizer, model, batch_size=8) :
    """Extract [CLS] token embeddings from BERT for a list of texts."""
    all_embeddings = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"  Extracting embeddings {i+1}-{min(i+batch_size, len(texts))} / {len(texts)}")

        encoding = tokenizer(
            batch,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoding)

        # Use [CLS] token embedding as sentence representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)

#------------------------- Main -------------------------------

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(base, "results", "metrics.csv")

    # Load data
    df = pd.read_csv(metrics_path, encoding="latin-1")
    print(f"Loaded {len(df)} rows from metrics.csv\n")

    # Split annotated and unannotated
    annotated = df[df["strategy"].notna() & (df["strategy"] != "")].copy()
    unannotated = df[df["strategy"].isna()  | (df["strategy"] == "")].copy()
    print(f"Manual annotations: {len(annotated)}")
    print(f"To predict: {len(unannotated)}\n")

    # Encode labels
    le = LabelEncoder()
    annotated["label_id"] = le.fit_transform(annotated["strategy"])
    print(f"Classes: {list(le.classes_)}\n")

    # Load BERT
    print(f"Loading {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    bert = BertModel.from_pretrained(MODEL_NAME).to(device)
    print("BERT loaded.\n")

    # Extract embeddings for annotated responses
    print("Extracting BERT embeddings for annotated responses...")
    X = get_bert_embeddings(annotated["response"].tolist(), tokenizer, bert)
    y = annotated["label_id"].values
    print(f"Embeddings shape: {X.shape}\n")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # Train SVM
    print("Training SVM classifier...")
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, random_state=SEED))
    svm.fit(X_train, y_train)
    print("SVM trained.\n")

    # Cross-validation on full annotated set
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(svm, X, y, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0
    )
    print(f"Test Accuracy: {acc:.4f}")
    print(f"\nClassification Report:\n{report}")

    # Save evaluation
    eval_path = os.path.join(base, "results", "classifier_svm_evaluation.txt")
    with open(eval_path, "w") as f:
        f.write(f"BERT + SVM Classifier Results\n")
        f.write(f"{'='*40}\n\n")
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} "
                f"(+/- {cv_scores.std():.4f})\n\n")
        f.write(f"Classification Report:\n{report}")
    print(f"\nEvaluation saved to {eval_path}")

    # Predict unannotated responses
    if len(unannotated) > 0:
        print(f"\nExtracting embeddings for {len(unannotated)} unannotated responses...")
        X_pred = get_bert_embeddings(unannotated["response"].tolist(), tokenizer, bert)
        pred_ids = svm.predict(X_pred)
        pred_labels = le.inverse_transform(pred_ids)
        df.loc[unannotated.index, "strategy"]        = pred_labels
        df.loc[unannotated.index, "strategy_source"] = "bert_svm_predicted"
        print("Predictions complete.")

    # Save updated CSV
    df.to_csv(metrics_path, index=False, encoding="utf-8")
    print(f"\nUpdated metrics.csv saved.")
    print(f"\n{'='*50}")
    print(f"Classification complete!")
    print(f"Manual annotations: {len(annotated)}")
    print(f"SVM predictions: {len(unannotated)}")
    print(f"CV Accuracy: {cv_scores.mean():.4f}")
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
