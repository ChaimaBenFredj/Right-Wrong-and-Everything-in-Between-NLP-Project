# This file Applies SHAP explainability to a Logistic Regression classifier trained on the 5 linguistic metrics to predict response strategy.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# ------------------------- Config --------------------------------------

FEATURES = [
    "hedging_score",
    "certainty_score",
    "moral_score",
    "response_length",
    "sentiment_compound",
]

FEATURE_LABELS = [
    "Hedging Score",
    "Certainty Score",
    "Moral Language",
    "Response Length",
    "Sentiment",
]

STRATEGY_COLORS = {
    "balanced": "#E99E1E",
    "conditional": "#41B55C",
    "direct": "#4a90d9",
}

FEATURE_COLORS = ["#4a90d9", "#E99E1E", "#41B55C", "#e05c5c", "#8172B2"]

plt.rcParams.update({
    "font.family":"serif",
    "font.size":11,
    "axes.titlesize":13,
    "axes.labelsize": 11,
    "figure.dpi":150,
})


# --------------------- Save helper ---------------------------

def save(fig, name, output_dir):
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {name}.png")


# ------------------------ Global feature importance chart-----------

def chart_shap_bar(mean_abs_shap, output_dir):
    # bar chart of mean absolute SHAP values across all classes.
    sorted_idx = np.argsort(mean_abs_shap)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        [FEATURE_LABELS[i] for i in sorted_idx],
        mean_abs_shap[sorted_idx],
        color=[FEATURE_COLORS[i] for i in sorted_idx],
        height=0.5,
    )

    for bar, val in zip(bars, mean_abs_shap[sorted_idx]):
        ax.text(
            bar.get_width() + 0.0005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9
        )

    ax.set_xlabel("Mean |SHAP Value| (averaged across strategy classes)")
    ax.set_title("Global Feature Importance for Strategy Prediction (SHAP)")
    ax.set_xlim(0, mean_abs_shap.max() * 1.25)
    sns.despine()
    save(fig, "09_shap_bar", output_dir)


# ------------------- Per-class beeswarm chart --------------------

def chart_shap_beeswarm(shap_array, X_scaled, classes, output_dir):
  
    # beeswarm subplot per strategy class.
    np.random.seed(42)
    n_classes = len(classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(14, 5))

    scatter_ref = None

    for ci, (cls, ax) in enumerate(zip(classes, axes)):
        shap_cls = shap_array[:, :, ci]                            # (n_samples, n_features)
        sorted_idx = np.argsort(np.abs(shap_cls).mean(axis=0))     

        for yi, fi in enumerate(sorted_idx):
            vals= shap_cls[:, fi]
            feat_val= X_scaled[:, fi]
            jitter= np.random.uniform(-0.18, 0.18, size=len(vals))

            sc = ax.scatter(
                vals,
                yi + jitter,
                c=feat_val,
                cmap="RdBu_r",
                alpha=0.65,
                s=18,
                vmin=-2, vmax=2,
            )
            scatter_ref = sc

        ax.axvline(x=0, color="gray", linewidth=1, linestyle="--", alpha=0.6)

        # Y-axis labels only on leftmost plot
        ax.set_yticks(range(len(FEATURES)))
        if ci == 0:
            ax.set_yticklabels(
                [FEATURE_LABELS[i] for i in sorted_idx], fontsize=9
            )
        else:
            ax.set_yticklabels([])

        color = STRATEGY_COLORS.get(cls, "#333333")
        ax.set_title(f'"{cls.capitalize()}"', fontsize=12,
                     fontweight="bold", color=color)
        ax.set_xlabel("SHAP Value", fontsize=9)

    # Shared colorbar
    cbar = fig.colorbar(scatter_ref, ax=axes[-1], fraction=0.046, pad=0.06)
    cbar.set_label("Feature Value\n(standardised)", fontsize=8)
    cbar.set_ticks([-2, 0, 2])
    cbar.set_ticklabels(["Low", "Mid", "High"])

    fig.suptitle(
        "SHAP Values per Strategy Class — Which Metrics Drive Each Strategy?",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    save(fig, "10_shap_beeswarm", output_dir)


# ----------------------------- Main -----------------------------------

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(base, "results", "metrics.csv")
    output_dir = os.path.join(base, "results", "charts")
    os.makedirs(output_dir, exist_ok=True)

    # Loading data 
    df = pd.read_csv(metrics_path, encoding="latin-1")
    df = df[df["strategy"].notna() & (df["strategy"] != "")].copy()
    print(f"Rows with strategy labels: {len(df)}")
    print(f"Strategy distribution:\n{df['strategy'].value_counts().to_string()}\n")

    X = df[FEATURES].values
    y = df["strategy"].values

    # Encoding labels 
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)
    print(f"Classes: {classes}\n")

    # Scaling features 
    scaler= StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Training Logistic Regression 
    lr = LogisticRegression(
        max_iter=1000, random_state=42, C=1.0
    )
    lr.fit(X_scaled, y_enc)

    # Cross-validation
    cv = cross_val_score(lr, X_scaled, y_enc, cv=5, scoring="accuracy")
    print(f"5-fold CV Accuracy: {cv.mean():.4f} (+/- {cv.std():.4f})")

    # Classification report
    y_pred = lr.predict(X_scaled)
    print(f"\nClassification Report:\n")
    print(classification_report(y_enc, y_pred, target_names=classes, zero_division=0))

    # SHAP 
    #Computing SHAP values
    explainer = shap.LinearExplainer(lr, X_scaled,
                                       feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_scaled)

    # Normalisinf to consistent shape
    # LinearExplainer returns (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_array = np.array(shap_values)          
        shap_array = np.transpose(shap_array, (1, 2, 0))  
    else:
        shap_array = shap_values                    

    print(f"SHAP array shape: {shap_array.shape}  "
          f"(samples × features × classes)\n")

    # Global feature importance: mean |SHAP| over samples and classes 
    mean_abs_shap = np.abs(shap_array).mean(axis=(0, 2))

    # printing the summary 
    print("=" * 50)
    print("Global Feature Importance (mean |SHAP|):")
    for i in np.argsort(mean_abs_shap)[::-1]:
        print(f"  {FEATURE_LABELS[i]:<20}: {mean_abs_shap[i]:.5f}")

    print("\nPer-class Feature Importance:")
    for ci, cls in enumerate(classes):
        per_class = np.abs(shap_array[:, :, ci]).mean(axis=0)   
        top = np.argmax(per_class)
        print(f"  {cls:<14} → top feature: {FEATURE_LABELS[top]} "
              f"({per_class[top]:.5f})")

    # Generating charts
    chart_shap_bar(mean_abs_shap, output_dir)
    chart_shap_beeswarm(shap_array, X_scaled, classes, output_dir)

    print(f"\n{'='*50}")
    print(f"SHAP analysis complete!")
    print(f"CV Accuracy : {cv.mean():.4f}")


if __name__ == "__main__":
    main()
