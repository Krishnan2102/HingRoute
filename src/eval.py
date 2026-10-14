# 1. Imports: numpy for resampling, sklearn for metrics, matplotlib/seaborn for plots
import numpy as np
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,f1_score)
import matplotlib.pyplot as plt
import os
from typing import Any,Callable,Dict,List,Optional,Tuple
import pandas as pd
# 2. Function: compute_bootstrap_ci(y_true, y_pred, metric_fn, n_bootstraps=1000, seed=42)
#    - For i in range(n_bootstraps):
#         - Draw random indices with replacement (same length as dataset)
#         - Compute metric on this resample
#    - Return 2.5th and 97.5th percentiles
def compute_bootstrap_ci(
        y_true:np.ndarray,
        y_pred:np.ndarray,
        metric_fn: Callable[[np.ndarray,np.ndarray],float],
        n_bootstraps: int=1000,
        seed: int =42,
        alpha: float=0.05,
        ) -> Tuple[float,float]:
    """
    Computes a non-parametric bootstrap confidence interval for a given metric.
    """
    rng=np.random.default_rng(seed)
    n_samples=len(y_true)
    bootstrap_scores: List[float]=[]

    for _ in range(n_bootstraps):
        sample_indices=rng.integers(0,n_samples,size=n_samples)
        score=metric_fn(y_true[sample_indices],y_pred[sample_indices])
        bootstrap_scores.append(score)

    lower_pct=100*(alpha/2.0)
    upper_pct=100*((1-alpha)/2.0)
    ci_lower=float(np.percentile(bootstrap_scores,lower_pct))
    ci_upper=float(np.percentile(bootstrap_scores,upper_pct))

    return ci_lower,ci_upper

def plot_confusion_matrix(
        cm:np.ndarray,
        labels: List[str],
        save_path:str,
        title:str="Confusion Matrix",
        )->None:
    """
    Renders and saves a readable confusion matrix heatmap using matplotlib.
    """
    fig,ax=plt.subplots(figsize=(10,8))
    cax=ax.matshow(cm,cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels,rotation=45,ha="left",fontsize=9)
    ax.set_yticklabels(labels,fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val=cm[i,j]
            color="white" if val>(cm.max()/2.0) else "black"
            ax.text(j,i,str(val),ha="center",va="center",color=color,fontsize=8)

    ax.set_x_label("Predicted Label",fontweight="bold",labelpad=10)
    ax.set_ylabel("True Label",fontweight="bold")
    ax.set_title(title,pad=20,fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    plt.savefig(save_path,dpi=200)
    plt.close()    

# 3. Function: evaluate_predictions(y_true, y_pred, labels=None, output_dir=None)
#    - Compute accuracy
#    - Compute macro-F1
#    - Compute bootstrap 95% CI on macro-F1
#    - Generate per-class classification report
#    - Compute confusion matrix
#    - Save confusion matrix as CSV and heat-map plot (PNG)
#    - Return structured dict of results

def evaluate_predictions(
        y_true:List[str] | np.ndarray,
        y_pred: List[str] | np.ndarray,
        labels: Optional[List[str]]=None,
        output_prefix: Optional[str]=None,
)->Dict[str,Any]:
    """
    Computes all standard metrics: accuracy,macro-F1,bootstrap CI,per-class reports, 
    and optional confusion matrix artifacts. 
    """
    y_true_arr=np.asarray(y_true)
    y_pred_arr=np.asarray(y_pred)

    if labels is None:
        labels=sorted(list(set(y_true_arr)| set(y_pred_arr)))

    acc=float(accuracy_score(y_true_arr,y_pred_arr))
    macro_f1=float(f1_score(y_true_arr,y_pred_arr,average="macro",zero_division=0))

    ci_lower,ci_upper=compute_bootstrap_ci(
        y_true_arr,
        y_pred_arr,
        metric_fn=lambda yt,yp:float(f1_score(yt,yp,average="macro",zero_division=0)),
    )

    report_dict=classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    cm=confusion_matrix(y_true_arr,y_pred_arr,labels=labels)

    if output_prefix:
        cm_df=pd.DataFrame(cm,index=labels,columns=labels)
        csv_path=f"{output_prefix}_confusion_matrix.csv"
        png_path=f"{output_prefix}_confusion_matrix.png"

        cm_df.to_csv(csv_path)
        plot_confusion_matrix(cm,labels,png_path,title=f"Confusion Matix ({os.path.basename(output_prefix)})")

    return{
        "accuracy":acc,
        "macro_f1":macro_f1,
        "macro_f1_ci_95":(ci_lower,ci_upper),
        "classification_report":report_dict,
        "confusion_matrix":cm,
        "labels":labels,
        }