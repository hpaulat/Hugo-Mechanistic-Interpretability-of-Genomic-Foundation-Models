import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_layer_performance(
    roc_list,
    precision_list,
    recall_list,
    title="Layer-wise Probing Performance",
):
    """
    Each list should be a list of tuples: (layer, mean, std)
      - roc_list:       [(layer, auc_mean, auc_std), ...]
      - precision_list: [(layer, pre_mean, pre_std), ...]
      - recall_list:    [(layer, rec_mean, rec_std), ...]
    All three lists should contain the same set of layers.
    """

    def _to_series(results_list):
        results_sorted = sorted(results_list, key=lambda x: x[0])
        layers = [x[0] for x in results_sorted]
        means  = [x[1] for x in results_sorted]
        stds   = [x[2] for x in results_sorted]
        return layers, means, stds

    # Parse lists
    roc_layers, roc_means, roc_stds = _to_series(roc_list)
    pre_layers, pre_means, pre_stds = _to_series(precision_list)
    rec_layers, rec_means, rec_stds = _to_series(recall_list)

    # Basic sanity check
    if not (roc_layers == pre_layers == rec_layers):
        raise ValueError(
            "Layer mismatch: roc_list, precision_list, and recall_list must contain the same layers "
            "in the same set (ordering doesn't matter)."
        )

    layers = roc_layers

    plt.figure(figsize=(10, 6))

    # ROC-AUC
    plt.errorbar(
        layers, roc_means, yerr=roc_stds,
        marker='o', capsize=5, linestyle='-', linewidth=2,
        label="ROC-AUC"
    )

    # Precision
    plt.errorbar(
        layers, pre_means, yerr=pre_stds,
        marker='s', capsize=5, linestyle='-', linewidth=2,
        label="Precision"
    )

    # Recall
    plt.errorbar(
        layers, rec_means, yerr=rec_stds,
        marker='^', capsize=5, linestyle='-', linewidth=2,
        label="Recall"
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Model Layer", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(layers)
    plt.ylim(0.0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Baseline reference line at 0.5 (useful for ROC-AUC; less meaningful for precision/recall)
    plt.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)

    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.show()


