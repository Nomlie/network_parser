python - <<'PY'
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

pred_path = "/Users/nmfuphicsir.co.za/Documents/pHDProject/Results/ALL_with_global/chi2_fdr/hierarchy_testing_AMR/query_results/query_predictions.csv"
meta_path = "/Users/nmfuphicsir.co.za/Documents/pHDProject/Data/AFRO_TB/metadata/AFRO_dataset_meta_with_test_hierarchy.csv"

pred = pd.read_csv(pred_path)
meta = pd.read_csv(meta_path)

def norm(x):
    return str(x).strip().replace("-", "_")

# Guess sample column
sample_cols = ["sample_id", "Sample_ID", "sample", "Sample", "id", "ID"]
pred_sid = next(c for c in sample_cols if c in pred.columns)
meta_sid = next(c for c in sample_cols if c in meta.columns)

pred[pred_sid] = pred[pred_sid].map(norm)
meta[meta_sid] = meta[meta_sid].map(norm)

df = pred.merge(meta, left_on=pred_sid, right_on=meta_sid, how="inner")

rows = []

for parent in sorted(df["Lineage"].dropna().astype(str).unique()):
    sub = df[df["Lineage"].astype(str) == parent].copy()

    for true_col, pred_col in [
        ("Resistance_Profile", "predicted_level2"),
        ("AMR_binary", "predicted_level3"),
    ]:
        if true_col not in sub.columns or pred_col not in sub.columns:
            continue

        tmp = sub[[true_col, pred_col]].dropna().copy()
        tmp[true_col] = tmp[true_col].astype(str).str.strip().str.replace("-", "_", regex=False)
        tmp[pred_col] = tmp[pred_col].astype(str).str.strip().str.replace("-", "_", regex=False)

        if tmp.empty or tmp[true_col].nunique() < 2:
            rows.append({
                "parent_lineage": parent,
                "child_label": true_col,
                "n_samples": len(tmp),
                "n_true_classes": tmp[true_col].nunique(),
                "balanced_accuracy": None,
                "macro_true_positive_rate": None,
                "macro_f1": None,
                "recommended_route": "insufficient_support",
            })
            continue

        ba = balanced_accuracy_score(tmp[true_col], tmp[pred_col])
        macro_tpr = recall_score(tmp[true_col], tmp[pred_col], average="macro", zero_division=0)
        macro_f1 = f1_score(tmp[true_col], tmp[pred_col], average="macro", zero_division=0)

        if len(tmp) < 2 or tmp[true_col].nunique() < 2:
            route = "insufficient_support"
        elif true_col == "Resistance_Profile" and ba < 0.70:
            route = "fallback_to_binary_endpoint"
        else:
            route = "use_exact_child_prediction"

        rows.append({
            "parent_lineage": parent,
            "child_label": true_col,
            "n_samples": len(tmp),
            "n_true_classes": tmp[true_col].nunique(),
            "n_pred_classes": tmp[pred_col].nunique(),
            "balanced_accuracy": ba,
            "macro_true_positive_rate": macro_tpr,
            "macro_f1": macro_f1,
            "recommended_route": route,
        })

out = pd.DataFrame(rows)
print(out.sort_values(["child_label", "balanced_accuracy"], na_position="first").to_string(index=False))
PY