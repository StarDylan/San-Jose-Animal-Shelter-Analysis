from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
import plotly.graph_objects as go

# pyright: reportUnknownVariableType=none, reportUnknownArgumentType=none


def plot_roc_curves(y_true, scores_dict, out_html="roc_curve.html"):
    fig = go.Figure()
    for name, y_score in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        auc = roc_auc_score(y_true, y_score[:, 1])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             name="Chance", line=dict(dash="dash")))
    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        legend_title_text=None,
        width=900, height=550
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def plot_pr_curves(y_true, scores_dict, out_html="pr_curve.html"):
    fig = go.Figure()
    pos_rate = (sum(y_true) / len(y_true)) if len(y_true) else 0.0
    for name, y_score in scores_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score[:, 1])
        ap = average_precision_score(y_true, y_score[:, 1])
        _ = fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                                 name=f"{name} (AP={ap:.3f})"))
    _ = fig.add_trace(go.Scatter(x=[0,1], y=[pos_rate, pos_rate], mode="lines",
                             name=f"Baseline (pos rate={pos_rate:.3f})",
                             line=dict(dash="dash")))
    fig.update_layout(
        title="Precision–Recall Curve Comparison",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        legend_title_text=None,
        width=900, height=550
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def print_report(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba[:, 1])
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {name} ---")
    print("Confusion Matrix [TN FP; FN TP]:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
