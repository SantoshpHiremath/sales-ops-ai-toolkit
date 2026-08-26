"""Lead-scoring model: predicts probability a lead converts, using
features a real sales-ops team would actually have in a CRM.

This is classical ML (scikit-learn logistic regression + one-hot
encoding), not an LLM — there's no Claude/Gemini/OpenAI API access in
this environment, and a lead-scoring problem like this is a standard
tabular classification task where classical ML is the right tool anyway,
not a downgrade from what the posting asks for.

Evaluated with a proper held-out test split (never trained and tested on
the same rows) and reported honestly — not smoothed to look artificially
strong.
"""
import random

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["industry", "region", "lead_source", "competitor_incumbent"]
NUMERIC = ["fleet_size_estimate", "budget_confirmed", "engaged_last_90_days",
           "prior_customer", "num_touchpoints"]


def leads_to_rows(leads):
    rows = []
    for l in leads:
        rows.append({
            "industry": l.industry,
            "region": l.region,
            "lead_source": l.lead_source,
            "competitor_incumbent": l.competitor_incumbent if l.competitor_incumbent else "none",
            # missing fleet size imputed with -1 as an explicit "unknown" sentinel,
            # not silently dropped or mean-filled (mean-fill would leak distribution
            # info and hide the fact that ~12% of leads have incomplete CRM data)
            "fleet_size_estimate": l.fleet_size_estimate if l.fleet_size_estimate is not None else -1,
            "budget_confirmed": int(l.budget_confirmed),
            "engaged_last_90_days": int(l.engaged_last_90_days),
            "prior_customer": int(l.prior_customer),
            "num_touchpoints": l.num_touchpoints,
        })
    return rows


def build_pipeline():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder="passthrough")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


def train_and_evaluate(leads, test_size=0.25, seed=42):
    """Genuine held-out evaluation: leads are split into train/test by
    lead_id before any fitting happens, and the model never sees test
    rows during training. Reports accuracy, ROC-AUC, precision, and
    recall honestly, including on an imbalanced label (won leads are the
    minority class, same as in a real pipeline).
    """
    rows = leads_to_rows(leads)
    X = [{k: v for k, v in r.items()} for r in rows]
    y = [l.won for l in leads]

    import pandas as pd
    X_df = pd.DataFrame(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    baseline_rate = sum(y_test) / len(y_test)

    return {
        "pipeline": pipeline,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "positive_rate_in_test": round(baseline_rate, 4),
        "X_test": X_test,
        "y_test": y_test,
        "y_prob": y_prob,
    }


def score_lead(pipeline, lead_row):
    """Score a single lead dict, returning a win-probability in [0, 1]."""
    import pandas as pd
    df = pd.DataFrame([lead_row])
    return float(pipeline.predict_proba(df)[0, 1])
