"""P667 — NLP Sentiment Analysis

Streamlit app for binary sentiment classification, built around a Logistic
Regression classifier trained on TF-IDF features (see P667_02_Model_Training.ipynb).
The model was trained on 1,427 Amazon India product reviews — the app generalizes
to any short English review text in similar style.

Deployed via Streamlit Community Cloud from a GitHub repository.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# Page configuration — must be the first streamlit call
# =============================================================================
st.set_page_config(
    page_title="P667 NLP Sentiment Analysis",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Text preprocessing — MUST match notebook 1's clean_text() exactly
# =============================================================================
# Copy-pasted from P667_01 rather than imported — the deployment artifact has
# to be self-contained, and version-skew on a notebook import would be a
# silent disaster (predictions would shift in ways that are hard to debug).
def clean_text(text: str) -> str:
    """Apply the same cleaning the model was trained on. Order matters."""
    text = re.sub(r"The media could not be loaded\.\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)              # HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)    # URLs
    text = re.sub(r"#", "", text)                     # hash symbol
    text = re.sub(r"[^\x00-\x7F]+", " ", text)       # non-ASCII (emojis, etc.)
    text = re.sub(r"\s+", " ", text)                  # collapse whitespace
    return text.strip()


# =============================================================================
# Load model artifacts — cached so the pickle only deserializes once per session
# =============================================================================
@st.cache_resource
def load_artifacts():
    """Load the pickle and return the binary configuration only.

    The pickle contains both binary and three-class models, but for deployment
    we use binary only — picked deliberately for higher macro F1 (0.857) and
    cleaner interpretability.
    """
    artifact_path = Path(__file__).parent / "p667_artifacts.pkl"
    with open(artifact_path, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts["binary"]


@st.cache_data
def load_sample_reviews():
    """Load the bundled sample reviews CSV (real test-set examples)."""
    sample_path = Path(__file__).parent / "sample_reviews.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path)
    return pd.DataFrame()  # graceful fallback if the file is missing


# Load once, use everywhere
config = load_artifacts()
samples_df = load_sample_reviews()


# =============================================================================
# Prediction logic — the core of the app
# =============================================================================
def predict_one(review_text: str) -> dict:
    """Predict sentiment for a single review with confidence + top contributors.

    Returns a dict with: cleaned_text, label, confidence, top_positive_words,
    top_negative_words. The contributing-words logic is what makes this app
    feel different from a black-box classifier — LogReg's coefficients let us
    show *which words* drove the prediction.
    """
    cleaned = clean_text(review_text)

    if not cleaned:
        return {
            "cleaned_text": "",
            "label": "unknown",
            "confidence": 0.0,
            "error": "Review is empty after cleaning. Try a longer text.",
            "top_positive_words": [],
            "top_negative_words": [],
        }

    # Vectorize and predict
    X = config["vectorizer"].transform([cleaned])
    pred = config["model"].predict(X)[0]
    proba = config["model"].predict_proba(X)[0]

    label = config["label_map"][pred]
    confidence = float(proba.max())

    # Pull out which words drove this prediction
    # Each non-zero TF-IDF feature × its model coefficient = its contribution
    # to the positive logit. Sort to find top contributors in each direction.
    feature_names = config["vectorizer"].get_feature_names_out()
    review_vec = X.toarray().flatten()
    coefs = config["model"].coef_[0]            # binary LogReg has shape (1, n_features)
    contributions = review_vec * coefs           # element-wise contribution per feature

    # Only consider features actually present in this review (non-zero TF-IDF)
    active_idx = np.where(review_vec > 0)[0]
    if len(active_idx) == 0:
        return {
            "cleaned_text": cleaned,
            "label": label,
            "confidence": confidence,
            "top_positive_words": [],
            "top_negative_words": [],
        }

    active_contribs = [(feature_names[i], contributions[i]) for i in active_idx]
    # Sort by contribution: most positive first, most negative first (separately)
    pos_words = sorted([c for c in active_contribs if c[1] > 0],
                        key=lambda x: x[1], reverse=True)[:5]
    neg_words = sorted([c for c in active_contribs if c[1] < 0],
                        key=lambda x: x[1])[:5]

    return {
        "cleaned_text": cleaned,
        "label": label,
        "confidence": confidence,
        "top_positive_words": pos_words,
        "top_negative_words": neg_words,
    }


def predict_batch(reviews: list[str]) -> pd.DataFrame:
    """Predict sentiment for a list of reviews — returns a DataFrame."""
    cleaned = [clean_text(r) for r in reviews]
    # Drop empty rows — they'd error in the vectorizer
    valid = [(i, c) for i, c in enumerate(cleaned) if c]
    if not valid:
        return pd.DataFrame()

    valid_idx, valid_cleaned = zip(*valid)
    X = config["vectorizer"].transform(valid_cleaned)
    preds = config["model"].predict(X)
    probas = config["model"].predict_proba(X).max(axis=1)

    return pd.DataFrame({
        "review": [reviews[i] for i in valid_idx],
        "predicted_sentiment": [config["label_map"][p] for p in preds],
        "confidence": probas.round(3),
    })


# =============================================================================
# Sidebar — model metadata + how-to guide
# =============================================================================
with st.sidebar:
    st.markdown("## 📊 About this model")
    st.markdown(f"""
    **Model:** {config['model_name']}
    **Task:** Binary sentiment (positive / negative)
    **Macro F1:** {config['macro_f1']:.3f}
    **Features:** TF-IDF on cleaned review text
    """)

    st.markdown("---")
    st.markdown("## 📖 How to read predictions")
    st.markdown("""
    - **Label** — the model's best guess (positive or negative)
    - **Confidence** — the model's predicted probability for that label.
      Values near 0.5 mean the model is unsure.
    - **Top contributing words** — which words from your review pushed the
      prediction toward positive (green) or negative (red).
    """)

    st.markdown("---")
    st.markdown("## ⚠️ Limitations")
    st.markdown("""
    - Trained on **1,427 Amazon India product reviews** — performance may
      drop on reviews from other regions or product categories.
    - Only **binary** classification — neutral reviews are forced into either
      positive or negative.
    - 3-star reviews were excluded from training, so genuinely ambiguous
      reviews may produce low-confidence predictions.
    """)

    st.markdown("---")
    st.caption(
        "P667 Group 2 · Mentor: Sadiya Ansari · "
        "[GitHub repo](https://github.com/monsrock2024)"
    )


# =============================================================================
# Main page — header + tabs
# =============================================================================
st.title("📝 NLP Sentiment Analysis")
st.markdown(
    "Binary sentiment classifier trained on **1,427 Amazon product reviews** "
    "using TF-IDF features and Logistic Regression. Try a single review, run "
    "a batch from a CSV, or read about the model below."
)

tab_single, tab_batch, tab_about = st.tabs([
    "✏️ Single Review", "📁 Batch Mode", "ℹ️ About the Model"
])


# -----------------------------------------------------------------------------
# TAB 1 — Single review prediction
# -----------------------------------------------------------------------------
with tab_single:
    # Initialize the textbox's session_state key once so we can write to it from
    # the "Use this sample" button. The key here MUST match the text_area key=
    # below — Streamlit binds widget state to that key automatically.
    if "review_textarea" not in st.session_state:
        st.session_state["review_textarea"] = ""

    col_input, col_sample = st.columns([3, 1])

    with col_sample:
        st.markdown("**Try a sample**")
        st.caption("Pick a real review from the test set:")
        if not samples_df.empty:
            sample_choice = st.selectbox(
                "Sample reviews",
                options=range(len(samples_df)),
                format_func=lambda i: f"{samples_df.iloc[i]['rating']}-star: "
                                       f"{samples_df.iloc[i]['review'][:40]}...",
                label_visibility="collapsed",
            )
            if st.button("Use this sample", use_container_width=True):
                # Write directly to the widget's key — this is the only reliable
                # way to update a Streamlit text_area from another widget. The
                # rerun forces an immediate refresh so the textbox shows the
                # new value without waiting for the next user interaction.
                st.session_state["review_textarea"] = samples_df.iloc[sample_choice]["review"]
                st.rerun()

    with col_input:
        # No value= argument — the widget reads from st.session_state["review_textarea"]
        # because key= is set. Setting both value= and key= causes Streamlit to
        # silently ignore programmatic updates after the first interaction.
        review = st.text_area(
            "Paste a product review here:",
            height=160,
            placeholder="e.g., 'Battery life is amazing and the camera quality "
                        "exceeded my expectations...'",
            key="review_textarea",
        )

        analyze_clicked = st.button("🔍 Analyze sentiment", type="primary",
                                     use_container_width=True)

    # Run prediction when button is clicked OR when sample was just loaded
    if analyze_clicked and review.strip():
        result = predict_one(review)

        if result.get("error"):
            st.warning(result["error"])
        else:
            # Headline result — large colored badge
            label = result["label"]
            confidence = result["confidence"]
            color = "#2e7d32" if label == "positive" else "#c62828"
            st.markdown(
                f"""<div style='padding: 20px; border-radius: 8px;
                background-color: {color}; color: white; text-align: center;
                margin: 20px 0;'>
                <h2 style='margin: 0; color: white;'>
                Prediction: {label.upper()}
                </h2>
                <p style='margin: 8px 0 0 0; font-size: 1.1em;'>
                Confidence: {confidence:.1%}
                </p>
                </div>""",
                unsafe_allow_html=True,
            )

            # Confidence interpretation
            if confidence < 0.6:
                st.info(
                    "⚠️ Low confidence — the model is unsure. This often happens "
                    "with mixed-sentiment reviews (e.g., '3-star' style reviews "
                    "where the reviewer praises some features and criticizes others)."
                )

            # Top contributing words — the interpretability win
            st.markdown("### What drove this prediction?")
            st.caption(
                "Words from your review that pushed the model's decision in each "
                "direction. Larger values = stronger contribution."
            )

            col_pos, col_neg = st.columns(2)

            with col_pos:
                st.markdown("**🟢 Pushed toward positive**")
                if result["top_positive_words"]:
                    pos_df = pd.DataFrame(
                        result["top_positive_words"], columns=["word", "contribution"]
                    )
                    pos_df["contribution"] = pos_df["contribution"].round(3)
                    st.dataframe(pos_df, hide_index=True, use_container_width=True)
                else:
                    st.caption("No positive contributors in this review.")

            with col_neg:
                st.markdown("**🔴 Pushed toward negative**")
                if result["top_negative_words"]:
                    neg_df = pd.DataFrame(
                        result["top_negative_words"], columns=["word", "contribution"]
                    )
                    neg_df["contribution"] = neg_df["contribution"].round(3)
                    st.dataframe(neg_df, hide_index=True, use_container_width=True)
                else:
                    st.caption("No negative contributors in this review.")

            # Show what the model actually saw, for transparency
            with st.expander("📋 See cleaned text (what the model actually saw)"):
                st.code(result["cleaned_text"], language=None)

    elif analyze_clicked:
        st.warning("Please paste a review first.")


# -----------------------------------------------------------------------------
# TAB 2 — Batch mode (CSV upload)
# -----------------------------------------------------------------------------
with tab_batch:
    st.markdown("### Analyze multiple reviews at once")
    st.caption(
        "Upload a CSV with a column named `review` (or `body`), or use the "
        "bundled sample CSV. Predictions and confidences will appear in a "
        "downloadable table below."
    )

    col_upload, col_sample_btn = st.columns([3, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "CSV file",
            type=["csv"],
            help="The file must have a column named 'review' or 'body' "
                 "containing the text to analyze.",
        )

    with col_sample_btn:
        st.markdown("&nbsp;")  # spacer to align with the uploader
        use_sample = st.button("Use bundled sample", use_container_width=True)

    # Load reviews from upload or sample
    batch_df = None
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            text_col = next(
                (c for c in ["review", "body", "text"] if c in batch_df.columns),
                None,
            )
            if text_col is None:
                st.error(
                    "CSV must have a column named 'review', 'body', or 'text'. "
                    f"Found: {list(batch_df.columns)}"
                )
                batch_df = None
            else:
                batch_df = batch_df.rename(columns={text_col: "review"})
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    elif use_sample and not samples_df.empty:
        batch_df = samples_df.copy()

    # Run batch prediction if we have data
    if batch_df is not None and len(batch_df) > 0:
        with st.spinner(f"Analyzing {len(batch_df)} reviews..."):
            results = predict_batch(batch_df["review"].astype(str).tolist())

        if results.empty:
            st.warning("No reviews could be analyzed — they may all be empty after cleaning.")
        else:
            # Summary metrics row
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total analyzed", len(results))
            col_b.metric(
                "Positive", f"{(results['predicted_sentiment'] == 'positive').sum()}"
            )
            col_c.metric(
                "Negative", f"{(results['predicted_sentiment'] == 'negative').sum()}"
            )

            # Truncate long reviews for display, keep full text for download
            display_df = results.copy()
            display_df["review"] = display_df["review"].apply(
                lambda r: r[:120] + "..." if len(r) > 120 else r
            )
            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "confidence": st.column_config.ProgressColumn(
                        "confidence", min_value=0.5, max_value=1.0, format="%.3f"
                    ),
                },
            )

            # Download full results
            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download full results as CSV",
                data=csv_bytes,
                file_name="p667_predictions.csv",
                mime="text/csv",
            )


# -----------------------------------------------------------------------------
# TAB 3 — About the model
# -----------------------------------------------------------------------------
with tab_about:
    st.markdown("## How this model was built")

    st.markdown("""
    This is the deployment of a binary sentiment classifier built as part of
    project **P667 — NLP Sentiment Analysis** (DS Group 2, mentored
    by Sadiya Ansari). The full development happened across two notebooks:
    """)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Pipeline")
        st.markdown(f"""
        1. **Data:** 1,440 Amazon India product reviews (title, rating, body)
        2. **Cleaning:** dropped 13 non-English reviews, stripped HTML/URLs/emoji,
           preserved original casing for downstream models
        3. **Labeling:** ratings 1-2 → negative, 4-5 → positive (3-star excluded)
        4. **Features:** TF-IDF with 1-2 word n-grams, 5,000 features max,
           sublinear TF scaling
        5. **Models compared:** LogReg, LinearSVC, MultinomialNB, ComplementNB,
           RandomForest — each with and without `class_weight='balanced'`
        6. **Winner:** **{config['model_name']}** (unweighted, body-only) — macro F1 **{config['macro_f1']:.3f}**
        """)

    with col_right:
        st.markdown("### Why these choices?")
        st.markdown("""
        - **Macro F1 not accuracy** — the three-class scheme tested separately
          showed several models scoring 0% F1 on neutrals while still hitting 76%
          accuracy. Macro F1 punishes that failure mode.
        - **Logistic Regression** — three configurations tied at 0.857 F1.
          We picked LogReg/body/unweighted as the simplest of the three, and
          for its directly interpretable coefficients (the "top contributing
          words" feature in this app comes from those coefficients).
        - **Body only, not title+body** — Section 4 of the modeling notebook
          tested whether including review titles helped. It didn't, so we use
          body-only for cleaner deployment.
        - **No deep learning in v1** — DistilBERT fine-tuning needs GPU compute
          to do justice to. Deferred to a follow-up phase.
        """)

    st.markdown("---")

    st.markdown("### Known limitations")
    st.warning("""
    **Distribution shift:** The training data is Amazon India product reviews,
    with distinctive phrasing patterns ("first class", "value for money",
    Indian-English idioms). Reviews written in other regional varieties of
    English may produce less accurate or lower-confidence predictions. A
    production deployment would benefit from training data covering broader
    English-review styles.

    **Binary only:** This deployment classifies as positive or negative only.
    A three-class model (positive/neutral/negative) was also trained but
    scored only 0.69 macro F1, with the neutral class being the weak point
    (F1 of just 0.38). Binary was chosen for the deployment because it's the
    decision the data actually supports.
    """)
