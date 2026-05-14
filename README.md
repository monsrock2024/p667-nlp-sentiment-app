# P667 — NLP Sentiment Analysis (Streamlit App)

A web app for binary sentiment classification, built around a Logistic Regression
classifier trained on 1,427 Amazon product reviews. Built as the deployment
artifact for project P667 — DS Group 2, mentored by Sadiya Ansari.

**Live demo:** [p667-sentiment.streamlit.app](https://p667-nlp-sentiment-app-bewrugqg4pnewx3lwpeq2a.streamlit.app/) *(replace with your actual URL after deploy)*

## What it does

- **Single review mode** — paste any product review and get a positive/negative prediction with confidence and the words that drove the decision
- **Batch mode** — upload a CSV of reviews and get all predictions in one go, downloadable as CSV
- **About the model** — full transparency on training data, methodology, and known limitations

## How the model was built

| Stage | What happened |
|---|---|
| Data | 1,440 Amazon India product reviews → 1,427 after dropping non-English |
| Cleaning | HTML/URL stripping, emoji removal, casing preserved |
| Labeling | Binary (1-2★ = neg, 4-5★ = pos), 3-star reviews excluded from training |
| Features | TF-IDF with 1-2 word n-grams, 5,000 features, fit on training rows only |
| Compared | LogisticRegression, LinearSVC, MultinomialNB, ComplementNB, RandomForest |
| Winner | **Logistic Regression** — macro F1 0.857, simplest of three tied configs |

The full development is documented in two Jupyter notebooks
(`P667_01_EDA_Preprocessing.ipynb` and `P667_02_Model_Training.ipynb`) plus
companion walkthrough PDFs.

## File layout

```
streamlit-app/
├── app.py                  # main Streamlit app
├── requirements.txt        # pinned dependencies
├── p667_artifacts.pkl      # the trained model + vectorizer
├── sample_reviews.csv      # 10 real test reviews for the demo
├── .streamlit/
│   └── config.toml         # theme configuration
└── README.md               # this file
```

## Running locally

```bash
git clone https://github.com/monsrock2024/p667-nlp-sentiment-app.git
cd p667-sentiment-app
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deploying to Streamlit Community Cloud

1. Push this folder to a public GitHub repo (root of the repo = root of this folder).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Connect your GitHub account and select the repo.
4. Set the main file path to `app.py`.
5. Open **Advanced settings** and select **Python 3.13** (or 3.12). The pinned
   `scikit-learn==1.8.0` requires Python 3.11+, so don't accept the default if
   it's older.
6. Click **Deploy**.

First deploy takes ~3-5 minutes for `pip install`. Subsequent pushes auto-redeploy.

## Why these dependency versions

The `requirements.txt` pins exact versions matching the environment that
trained the model. Version mismatches (especially in scikit-learn) are the
single most common cause of "the pickle won't load" errors on deployment —
sklearn changes its internal class layouts between minor versions, so a
pickle saved with 1.8 may not load cleanly with 1.7.

If you need to upgrade dependencies later, retrain the model in the same
environment first and re-pickle.

## Limitations (in the app's About tab too)

- Trained on **Amazon India** product reviews only — accuracy may drop on
  reviews from other regions or product categories.
- **Binary only** — neutral reviews get forced into either positive or negative.
- 3-star reviews were excluded from training; ambiguous reviews may produce
  low-confidence predictions.

## Project team

- **Group 2:** Moin Mohammed, Kaveeshvar S, Vrushali P Kasliwal, Kushala R Kashyap, Shrinidhi Joshi, Khushi Sudarshan Choudhari
- **Mentor:** Sadiya Ansari
- **Institute:** ExcelR
