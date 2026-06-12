# This file Computes linguistic metrics on all 135 collected responses
# Metrics computed: response_length, hedging_score, moral_score, certainty_score, refusal_score, sentiment

import json
import os
import re
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------- Lexicons -------------------------------

HEDGING_WORDS = [
         "perhaps", "maybe", "might", "could", "possibly", "arguably",
         "it seems", "it appears", "in some cases", "it is difficult",
         "not straightforward", "debatable", "uncertain", "unclear",
         "it depends", "one might", "some would", "others might",
         "to some extent", "in certain contexts", "it is hard to say"
]

MORAL_WORDS = [
    "ethical", "unethical", "moral", "immoral", "wrong", "right",
    "just", "unjust", "fair", "unfair", "harm", "harmful", "duty",
    "dignity", "justice", "integrity", "responsibility", "obligation",
    "virtue", "compassion", "rights", "violation", "principle",
    "conscience", "equitable", "discriminatory", "exploitation"
]

CERTAINTY_WORDS = [
      "should", "must", "clearly", "obviously", "undoubtedly",
        "certainly", "definitely", "is wrong", "is right", "is unethical",
        "is ethical", "without question", "it is clear", "there is no doubt",
        "it is essential", "it is necessary", "unacceptable", "imperative"
]

REFUSAL_WORDS = [
             "i cannot", "i can't", "i am not able", "i'm not able",
           "i won't", "i will not", "it's not appropriate", "not appropriate",
           "i don't think it's my place", "as an ai", "i must clarify",
            "i should note", "i am unable", "this is a complex issue",
            "i'd rather not", "i prefer not"
]

# ------------------------ Metric functions ---------------------------

def word_count(text):
    return len(text.split())

def lexicon_score(text, lexicon):
    """Count lexicon matches per 100 words (normalized frequency)."""
    text_lower = text.lower()
    words = text.split()
    if not words:
        return 0.0
    matches = sum(1 for term in lexicon if term in text_lower)
    return round((matches / len(words)) * 100, 4)

def get_sentiment(text, analyzer):
    scores = analyzer.polarity_scores(text)
    compound = round(scores["compound"], 4)
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return compound, label

def has_refusal(text):
    text_lower = text.lower()
    return any(marker in text_lower for marker in REFUSAL_WORDS)

# ------------------- Main analysis -----------------------------

def analyze_responses(responses_path, output_path):
    with open(responses_path, "r", encoding="latin-1") as f:
        responses = json.load(f)

    analyzer = SentimentIntensityAnalyzer()

    rows = []
    for i, entry in enumerate(responses):
        text = entry["response"]

        sentiment_compound, sentiment_label = get_sentiment(text, analyzer)

        row = {
            # Identifiers
            "scenario_id": entry["scenario_id"],
            "domain":  entry["domain"],
            "model":  entry["model"],

            # Linguistic metrics
            "response_length": word_count(text),
            "hedging_score": lexicon_score(text, HEDGING_WORDS),
            "moral_score": lexicon_score(text, MORAL_WORDS),
            "certainty_score": lexicon_score(text, CERTAINTY_WORDS),
            "refusal_score": lexicon_score(text, REFUSAL_WORDS),
            "has_refusal": int(has_refusal(text)),

            # Sentiment
            "sentiment_compound": sentiment_compound,
            "sentiment_label": sentiment_label,

            # Raw response (for manual annotation)
            "response":  text,

            # Annotation columns (some of them to be filled in by hand and the rest with bert+svm)
            "strategy": "",   # direct, balanced, conditional
            "mentions_law": "",   # yes orno
            "mentions_religion": "",   # yes or no
            "takes_moral_stance": "",   # yes or no
            "acknowledges_complexity": "",  # yes pr no
        }

        rows.append(row)
        print(f"[{i+1}/{len(responses)}] Analyzed {entry['scenario_id']} / {entry['model']}")

    # Write to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*50}")
    print(f"Analysis complete!")
    print(f"{len(rows)} rows saved to {output_path}")

# --------------------- Main --------------------------

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analyze_responses(
        responses_path=os.path.join(base, "data", "responses.json"),
        output_path=os.path.join(base, "results", "metrics.csv")
    )
