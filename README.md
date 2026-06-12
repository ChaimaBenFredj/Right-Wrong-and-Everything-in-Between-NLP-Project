# Right, Wrong, and Everything in Between
### How Large Language Models Navigate Ethical Dilemmas

 **Project done by: Chaima Ben Fredj — Master's in Data Science for Economics**

**Natural Language Processing — Prof. Alfio Ferrara**

 
---
 
## Project Overview
 
This project examines how three large language models — LLaMA-3-70B, Qwen-2.5-72B, and DeepSeek-V3 — navigate morally ambiguous scenarios across medical, privacy, and fairness domains. Using a custom dataset of 45 ethical scenarios and 135 open-ended responses, we analyse linguistic patterns including hedging, certainty, moral vocabulary, sentiment, and rhetorical strategy. We additionally benchmark moral alignment using the ETHICS dataset and apply SHAP explainability to identify which linguistic metrics drive each rhetorical strategy.
 
---
 
## Repository Structure
 
```
├── data/
│   ├── scenarios.json          # 45 custom ethical scenarios (15 per domain)
│   ├── responses.json          # 135 LLM responses collected via HuggingFace API
│   └── ethics_sample.json      # 75 sampled scenarios from the ETHICS benchmark
│
├── results/
│   ├── charts/                 # All 10 generated figures (PNG)
│   ├── metrics.csv             # Linguistic metrics + strategy labels for all 135 responses
│   ├── ethics_results.csv      # Per-response ETHICS benchmark classification results
│   ├── ethics_summary.csv      # Accuracy and F1 per model per ETHICS subset
│   └── classifier_svm_evaluation.txt  # BERT+SVM classifier evaluation report
│
├── src/
│   ├── collect.py              # Queries 3 LLMs with the 45 scenarios via HuggingFace API
│   ├── analyze.py              # Computes 6 linguistic metrics on all responses
│   ├── classify_svm.py         # BERT + SVM strategy classifier (direct/balanced/conditional)
│   ├── shap_analysis.py        # SHAP explainability on linguistic metrics
│   ├── sample_ethics.py        # Samples balanced scenarios from the ETHICS benchmark
│   ├── ethics_eval.py          # Queries LLMs on ETHICS scenarios and evaluates accuracy
│   └── visualize.py            # Generates all charts and statistical tests
│
├── 01_data_collection.ipynb    # Notebook: scenario design and LLM response collection
├── 02_analysis.ipynb           # Notebook: linguistic metrics and ETHICS benchmark evaluation
├── 03_classification_and_shap.ipynb  # Notebook: strategy classification and SHAP explainability
├── 04_visualization_and_results.ipynb # Notebook: all charts and comprehensive results discussion
│
├── Chaima_BF_NLP_Report.pdf    # Final project report
└── README.md
```
 
---
 
## Pipeline
 
```
collect.py → responses.json
      ↓
analyze.py → metrics.csv
      ↓
classify_svm.py → strategy labels in metrics.csv
      ↓
shap_analysis.py → SHAP charts
      ↓
sample_ethics.py → ethics_sample.json
      ↓
ethics_eval.py → ethics_results.csv + ethics_summary.csv
      ↓
visualize.py → 10 charts
```
 
---
 
## Models Used
 
| Model | Developer | Parameters |
|---|---|---|
| Meta-Llama-3-70B-Instruct | Meta | 70B |
| Qwen2.5-72B-Instruct | Alibaba | 72B |
| DeepSeek-V3-0324 | DeepSeek AI | ~685B (MoE) |
 
---
 
## Requirements
 
A HuggingFace API key is required to run `collect.py` and `ethics_eval.py`.
Change the API key in `.env` file in the project root with your Key:
```
HF_API_KEY=your_key_here
```
 
---
 
## AI Usage Disclaimer
 
Parts of this project were developed with the assistance of Anthropic's Claude (claude-sonnet-4-6). The AI was used to support the development of the data collection pipeline, helping with the code development, and the drafting of descriptive texts. All AI-assisted content has been carefully reviewed, edited, and validated by the author. Full responsibility for the project's content, reasoning, and academic integrity rests with the author.
