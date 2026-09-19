import joblib, os
import matplotlib.pyplot as plt
import numpy as np

DATASETS = {
    'UCI':     'outputs/UCI_tau124',
    'TaFeng':  'outputs/TAFENG_tau39',
    'CDNOW':   'outputs/CDNOW_tau181',
}

FIGURES_DIR = 'outputs/paper_figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

def main():
    artifacts = {}
    for name, run_dir in DATASETS.items():
        meta_path = f"{run_dir}/models/pipeline_meta.pkl"
        if os.path.exists(meta_path):
            artifacts[name] = joblib.load(meta_path)
            
    fig, axes = plt.subplots(1, len(artifacts), figsize=(6*len(artifacts), 5), sharey=False)
    if len(artifacts) == 1:
        axes = [axes]

    for ax, (name, meta) in zip(axes, artifacts.items()):
        mc = meta.get('monte_carlo_results', {})
        if not mc:
            ax.text(0.5, 0.5, 'No MC data', ha='center', va='center')
            ax.set_title(name)
            continue

        w_arr = mc.get('weibull_profits_arr')
        r_arr = mc.get('rfm_profits_arr')
        lr_arr = mc.get('lr_profits_arr')

        if w_arr is not None and r_arr is not None:
            ax.hist(w_arr, bins=50, alpha=0.6, color='#3498db', label='Weibull', density=True)
            ax.hist(r_arr, bins=50, alpha=0.6, color='#e74c3c', label='RFM', density=True)
            
            if lr_arr is not None:
                ax.hist(lr_arr, bins=50, alpha=0.6, color='#9b59b6', label='LR+EVI', density=True)
                ax.axvline(np.median(lr_arr), color='#9b59b6', lw=2, ls='--')
                
            ax.axvline(np.median(w_arr), color='#3498db', lw=2, ls='--')
            ax.axvline(np.median(r_arr), color='#e74c3c', lw=2, ls='--')
            ax.axvline(0, color='#333', lw=0.8)

        ax.set_title(f'{name} — Profit Distribution', fontsize=11, fontweight='bold')
        ax.set_xlabel('Profit (MU)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Monte Carlo Profit Distribution (n=1,000 iterations)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(FIGURES_DIR, 'fig_mc_profit_distribution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved → {path}')

if __name__ == "__main__":
    main()
