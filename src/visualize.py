# this file Generates all charts and visualizations for the paper.

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ------------------- Config --------------------------

COLORS = {
    "llama3": "#4a90d9",
    "qwen": "#E99E1E",
    "deepseek":"#41B55C",
}

DOMAIN_COLORS = {
    "medical":"#e05c5c",
    "privacy": "#8172B2",
    "fairness":"#937860",
}

STRATEGY_COLORS = {
    "direct": "#4a90d9",
    "balanced": "#E99E1E",
    "conditional": "#41B55C",
}

MODEL_LABELS = {
    "llama3":"LLaMA-3-70B",
    "qwen": "Qwen-2.5-72B",
    "deepseek":"DeepSeek-V3",
}

DOMAIN_LABELS = {
    "medical":"Medical",
    "privacy":"Privacy",
    "fairness":"Fairness",
}

plt.rcParams.update({
    "font.family":"serif",
    "font.size":11,
    "axes.titlesize":13,
    "axes.labelsize":11,
    "figure.dpi": 150,
})


# -------------------- Helper----------------------------

def save(fig, name, output_dir):
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {name}.png")


# ----- Chart 1: Strategy distribution by model

def chart_strategy_by_model(df, output_dir):
    strategies = ["direct","balanced","conditional"]
    models = ["llama3","qwen","deepseek"]

    data = {}
    for model in models:
        subset = df[df["model"] == model]["strategy"].value_counts(normalize=True) * 100
        data[model] = [subset.get(s, 0) for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(models))
    bottom = [0] * len(models)

    for i, strategy in enumerate(strategies):
        values = [data[m][i] for m in models]
        bars = ax.bar(x, values, bottom=bottom,
                       color=STRATEGY_COLORS[strategy],
                       label=strategy.capitalize(), width=0.5)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Response Strategy Distribution by Model")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 110)
    sns.despine()
    save(fig, "01_strategy_by_model", output_dir)


# ----- Chart 2: Hedging score by model 

def chart_hedging_by_model(df, output_dir):
    models = ["llama3", "qwen", "deepseek"]
    means= [df[df["model"] == m]["hedging_score"].mean() for m in models]
    stds= [df[df["model"] == m]["hedging_score"].std()for m in models]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [MODEL_LABELS[m] for m in models], means,
        yerr=stds, capsize=5,
        color=[COLORS[m] for m in models],
        width=0.5, error_kw={"elinewidth": 1.5}
    )
    ax.set_ylabel("Hedging Score (per 100 words)")
    ax.set_title("Hedging Language Score by Model")

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    sns.despine()
    save(fig, "02_hedging_by_model", output_dir)


#------------- Chart 3: Moral language score by domain 

def chart_moral_by_domain(df, output_dir):
    models= ["llama3", "qwen", "deepseek"]
    domains = ["medical", "privacy", "fairness"]
    x= range(len(domains))
    width= 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        means = [df[(df["model"] == model) & (df["domain"] == d)]["moral_score"].mean()
                 for d in domains]
        ax.bar([xi + i*width for xi in x], means,
               width=width, color=COLORS[model],
               label=MODEL_LABELS[model])

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in domains])
    ax.set_ylabel("Moral Language Score (per 100 words)")
    ax.set_title("Moral Language Score by Domain and Model")
    ax.legend()
    sns.despine()
    save(fig, "03_moral_by_domain", output_dir)


# --------Chart 4: Sentiment distribution by model 

def chart_sentiment_by_model(df, output_dir):
    models = ["llama3", "qwen", "deepseek"]
    sentiments= ["positive", "negative"]
    sent_colors = {
        "positive": "#55A868",
        "negative": "#C44E52"
    }

    data = {}
    for model in models:
        subset = df[df["model"] == model]["sentiment_label"].value_counts(normalize=True) * 100
        data[model] = [subset.get(s, 0) for s in sentiments]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(models))
    bottom = [0] * len(models)

    for sentiment in sentiments:
        values = [data[m][sentiments.index(sentiment)] for m in models]
        ax.bar(x, values, bottom=bottom,
               color=sent_colors[sentiment],
               label=sentiment.capitalize(), width=0.5)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Sentiment Distribution by Model")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 110)
    sns.despine()
    save(fig, "04_sentiment_by_model", output_dir)


# ---------- Chart 5: Response length by model 

def chart_length_by_model(df, output_dir):
    models = ["llama3", "qwen", "deepseek"]
    data = [df[df["model"] == m]["response_length"].values for m in models]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})

    for patch, model in zip(bp["boxes"], models):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.7)

    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel("Word Count")
    ax.set_title("Response Length Distribution by Model")
    sns.despine()
    save(fig, "05_length_by_model", output_dir)


#----------- Chart 6: Certainty score by domain 

def chart_certainty_by_domain(df, output_dir):
    models = ["llama3", "qwen", "deepseek"]
    domains = ["medical", "privacy", "fairness"]
    x = range(len(domains))
    width= 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        means = [df[(df["model"] == model) & (df["domain"] == d)]["certainty_score"].mean()
                 for d in domains]
        ax.bar([xi + i*width for xi in x], means,
               width=width, color=COLORS[model],
               label=MODEL_LABELS[model])

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in domains])
    ax.set_ylabel("Certainty Score (per 100 words)")
    ax.set_title("Certainty Score by Domain and Model")
    ax.legend()
    sns.despine()
    save(fig, "06_certainty_by_domain", output_dir)


# ----------- Chart 7: Heatmap of all metrics by model 

def chart_heatmap(df, output_dir):
    models= ["llama3", "qwen", "deepseek"]
    metrics = ["response_length", "hedging_score", "moral_score",
               "certainty_score", "refusal_score", "sentiment_compound"]

    metric_labels = {
        "response_length":    "Response Length",
        "hedging_score":      "Hedging Score",
        "moral_score":        "Moral Language",
        "certainty_score":    "Certainty Score",
        "refusal_score":      "Refusal Score",
        "sentiment_compound": "Sentiment"
    }

    matrix = pd.DataFrame(index=[MODEL_LABELS[m] for m in models],
                          columns=[metric_labels[m] for m in metrics])

    for model in models:
        for metric in metrics:
            val = df[df["model"] == model][metric].mean()
            matrix.loc[MODEL_LABELS[model], metric_labels[metric]] = round(val, 4)

    matrix = matrix.astype(float)

    # Normalize each column for visualization
    normalized = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(normalized, annot=matrix.values, fmt=".3f",
                cmap="YlOrRd", ax=ax, linewidths=0.5,
                cbar_kws={"label": "Normalized Score"})
    ax.set_title("Linguistic Metrics Heatmap by Model (normalized)")
    plt.tight_layout()
    save(fig, "07_heatmap", output_dir)


# -------------- Statistical tests 

def run_statistical_tests(df, output_dir):
    models= ["llama3", "qwen", "deepseek"]
    metrics = ["hedging_score", "moral_score", "certainty_score", "response_length"]

    results = []
    for metric in metrics:
        groups = [df[df["model"] == m][metric].values for m in models]
        f_stat, p_value = stats.kruskal(*groups)
        results.append({
            "metric":  metric,
            "H_stat":  round(f_stat, 4),
            "p_value": round(p_value, 4),
            "significant": "Yes" if p_value < 0.05 else "No"
        })

    results_df = pd.DataFrame(results)
    path = os.path.join(output_dir, "statistical_tests.csv")
    results_df.to_csv(path, index=False)
    print(results_df.to_string(index=False))

 
# ----------- Chart 8: ETHICS benchmark accuracy by model and subset 
 
def chart_ethics_accuracy(ethics_df, output_dir):
    """
    Grouped bar chart showing accuracy per model per ETHICS subset.
    Each group of bars = one subset. Each bar = one model.
    """
    models = ["llama3", "qwen", "deepseek"]
    subsets = ["commonsense", "justice", "deontology"]
    x = range(len(subsets))
    width = 0.25
 
    fig, ax = plt.subplots(figsize=(9, 5))
 
    for i, model in enumerate(models):
        accs = []
        for subset in subsets:
            row = ethics_df[
                (ethics_df["model"] == model) &
                (ethics_df["subset"] == subset)
            ]
            accs.append(float(row["accuracy"].values[0]) if len(row) > 0 else 0.0)
 
        bars = ax.bar(
            [xi + i * width for xi in x],
            accs,
            width=width,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )
 
        # Value labels on top of each bar
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{acc:.2f}",
                    ha="center", va="bottom", fontsize=8
                )
 
    # Chance-level reference line
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1,
               label="Random chance (0.50)")
 
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(["Commonsense", "Justice", "Deontology"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("ETHICS Benchmark: Moral Alignment Accuracy by Model and Subset")
    ax.legend(loc="lower right")
    sns.despine()
    save(fig, "08_ethics_accuracy", output_dir)

# --------------- Main 

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(base, "results", "metrics.csv")
    output_dir= os.path.join(base, "results", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(metrics_path, encoding="latin-1")
    print(f"Loaded {len(df)} rows\n")
    #generating charts
    chart_strategy_by_model(df, output_dir)
    chart_hedging_by_model(df, output_dir)
    chart_moral_by_domain(df, output_dir)
    chart_sentiment_by_model(df, output_dir)
    chart_length_by_model(df, output_dir)
    chart_certainty_by_domain(df, output_dir)
    chart_heatmap(df, output_dir)

    #Running statistical tests
    run_statistical_tests(df, output_dir)
    ethics_path = os.path.join(base, "results", "ethics_summary.csv")
    if os.path.exists(ethics_path):
        ethics_df = pd.read_csv(ethics_path, encoding="utf-8")
        chart_ethics_accuracy(ethics_df, output_dir)
    else:
        print("ethics_summary.csv not found — skipping ethics chart.")

    print(f"\n{'='*50}")
    print(f"All charts saved")

if __name__ == "__main__":
    main()
