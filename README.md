# Mirror, Mirror on the Wall, Is AI Transforming Us All?
### RAISE-26 · Team: Agentic Minds

**Members**: Hanumath Mandadi · Abubaker Ahmadi  
**Affiliation**: Rutgers University–New Brunswick · Mercer County Community College

---

## Research Question

> **How is AI shaping the ways humans think, act, and interact?**

We answer this by analyzing **10,500 AI news headlines** through a six-stage open-weights NLP pipeline — covering semantic clustering, sentiment analysis, emotion profiling, and behavioral framing — without any proprietary APIs.

---

## Repository Contents

| File | Description |
|------|-------------|
| `AGENTIC_MINDS_RAISE_26_SUBMISSION_FINALS.ipynb` | Full analysis notebook |
| `dataset_A_news_full_10500_1.csv` | RAISE-26 Dataset A — 10,500 AI news headlines |

---

## Dataset

**RAISE-26 Dataset A** — 10,500 AI news headlines collected for the RAISE 2026 competition.

| Column | Description |
|--------|-------------|
| `title` | Headline text |
| `link` | Source URL |
| `date` | Publication date |
| `source` | News outlet |
| `classes_str` | Predefined behavioral class labels |
| `day_of_week`, `month`, `year`, `quarter` | Temporal features |
| `is_weekend` | Boolean weekend flag |
| `number_of_characters_title`, `number_of_words_title` | Headline length features |

---

## Pipeline Overview

1. **Environment Setup & Data Loading** — load the CSV, install dependencies
2. **Semantic Embedding** — `all-MiniLM-L6-v2` sentence embeddings
3. **Clustering** — KMeans with silhouette-optimized *k*, UMAP 3D visualization
4. **LLM Cluster Labeling** — open-weights LLM names each cluster
5. **Sentiment Analysis** — VADER per headline and cluster
6. **Behavioral Framing** — active vs. passive human agency detection
7. **Emotion Profiling** — NRC lexicon with negation handling across 8 emotions
8. **Convergent Validity** — ARI/NMI comparison against predefined `classes_str` labels

---

## Running the Notebook

### Option 1 — Google Colab (recommended, free GPU)

1. Open the notebook in Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Run all cells — the dataset loads automatically from the cloned repo

### Option 2 — Local

```bash
git clone https://github.com/Hanumath1506/headlines-under-the-microscope-nlp.git
cd headlines-under-the-microscope-nlp
pip install pandas numpy scikit-learn sentence-transformers transformers torch umap-learn plotly matplotlib tqdm nltk
jupyter notebook AGENTIC_MINDS_RAISE_26_SUBMISSION_FINALS.ipynb
```

---

## Key Findings

- **3 macro behavioral clusters** discovered via unsupervised learning: *Innovation & Commerce*, *Society & Ethics*, *Workforce & Human Agency*
- **~45% passive framing** — media portrays AI as acting *on* humans rather than being used *by* them
- **Trust dominates** emotional profiles across all clusters; Fear is highest in healthcare sub-topics
- **60-percentage-point gap** in negativity between most and least negative news sources
- Convergent validity (ARI/NMI) confirms discovered clusters align with predefined behavioral labels
