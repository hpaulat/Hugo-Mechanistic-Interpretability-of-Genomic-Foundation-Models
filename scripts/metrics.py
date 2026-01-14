import matplotlib.pyplot as plt

def plot_layer_performance(results_list, label="Delta Embeddings", title="ClinVar Pathogenicity Prediction per Layer"):
    plt.figure(figsize=(10, 6))
    
    plot_color = colors.get(label, '#333333') 

    results_sorted = sorted(results_list, key=lambda x: x[0])

    layers = [x[0] for x in results_sorted]
    means  = [x[1] for x in results_sorted]
    stds   = [x[2] for x in results_sorted]
        
    plt.errorbar(layers, means, yerr=stds, label=label, 
                 marker='o', capsize=5, linestyle='-', 
                 color=plot_color, linewidth=2)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Model Layer", fontsize=12)
    plt.ylabel("ROC-AUC", fontsize=12)
    plt.xticks(layers) 
    plt.ylim(0.4, 1.05) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', frameon=True)
    
    # Baseline reference
    plt.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, label='Random Chance')
    
    plt.tight_layout()
    plt.show()



def plot_umap_