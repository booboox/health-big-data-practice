"""Worksheet A: compare tree depths using the provided fixed split."""
from pathlib import Path
import json
import platform
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

DATA = Path(r"C:/Users/10840/Downloads/9. Ensemble/10. Ensemble/data/student")
OUT = Path(__file__).resolve().parent
SEED = 1010

def main():
    train = pd.read_csv(DATA / "train.csv")
    valid = pd.read_csv(DATA / "validation.csv")
    features = [c for c in train.columns if c not in ("sample_id", "malignant")]
    rows, predictions = [], []
    for depth in (1, 2, 3, 4, None):
        name = str(depth) if depth is not None else "unlimited"
        model = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
        model.fit(train[features], train["malignant"])
        row = {"max_depth": name, "actual_depth": model.get_depth(), "leaves": model.get_n_leaves()}
        for split, frame in (("train", train), ("validation", valid)):
            y = frame["malignant"]
            pred = model.predict(frame[features])
            prob = model.predict_proba(frame[features])[:, list(model.classes_).index(1)]
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            row.update({f"{split}_accuracy": accuracy_score(y, pred),
                        f"{split}_balanced_accuracy": balanced_accuracy_score(y, pred),
                        f"{split}_roc_auc": roc_auc_score(y, prob),
                        f"{split}_tn": int(tn), f"{split}_fp": int(fp),
                        f"{split}_fn": int(fn), f"{split}_tp": int(tp)})
            predictions.extend({"max_depth": name, "split": split, "sample_id": sid,
                                "malignant": int(actual), "prediction": int(p), "probability": float(q)}
                               for sid, actual, p, q in zip(frame["sample_id"], y, pred, prob))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tree_depth_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(OUT / "tree_depth_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT / "experiment_metadata.json").write_text(json.dumps({
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "random_state": SEED, "train_n": len(train), "validation_n": len(valid),
        "features": features, "parameters": "DecisionTreeClassifier defaults except max_depth and random_state",
        "test_set_used": False
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()
