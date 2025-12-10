import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from geostat_toolkit import (
    generate_synthetic_field, 
    run_kriging, 
    save_to_vtk, 
    save_grid_to_csv, 
    save_samples_to_csv
)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # 1. Load Config
    cfg = load_config(os.path.join(current_dir, 'demo_config.yaml'))
    dom = cfg['domain']

    # 2. Generate Truth
    print("Generating Synthetic Field...")
    x, y, truth = generate_synthetic_field(
        x_max=dom['x_max'], y_max=dom['y_max'],
        nx=dom['nx'], ny=dom['ny'],
        model_type=cfg['simulation']['model_type'],
        variance=cfg['simulation']['variance'],
        length_scale=cfg['simulation']['length_scale'],
        seed=cfg['simulation']['seed']
    )

    # 3. Sample
    print("Sampling...")
    rng = np.random.default_rng(cfg['sampling']['seed'])
    # Sample unique points
    flat_idx = rng.choice(dom['nx']*dom['ny'], size=cfg['sampling']['n_samples'], replace=False)
    idx_y, idx_x = np.divmod(flat_idx, dom['nx'])
    
    sx, sy = x[idx_x], y[idx_y]
    s_vals = truth[idx_y, idx_x]

    # 4. Kriging
    user_model = cfg['kriging'].get('variogram_model')
    params = None
    if user_model:
        params = {"model": user_model, "angle": cfg['kriging'].get('anisotropy_angle', 0.0)}
        print(f"Using manual parameters: {params}")
    
    pred, var, final_params = run_kriging(sx, sy, s_vals, x, y, params)
    print(f"Kriging done using: {final_params}")

    # 5. Export
    out_dir = cfg['output']['folder']
    os.makedirs(out_dir, exist_ok=True)
    
    print("Exporting Results...")
    
    # A. VTK Export (ParaView)
    save_to_vtk(
        os.path.join(out_dir, cfg['output']['vtk_filename']),
        {"Prediction": pred, "Variance": var, "GroundTruth": truth},
        x, y
    )
    
    # B. CSV Export (Results Grid)
    grid_csv_path = os.path.join(out_dir, "results_grid") # extension added automatically
    save_grid_to_csv(
        grid_csv_path,
        {
            "Ground_Truth": truth,
            "Prediction": pred,
            "Kriging_Variance": var
        },
        x, y
    )
    
    # C. CSV Export (Samples)
    samples_csv_path = os.path.join(out_dir, "samples")
    save_samples_to_csv(samples_csv_path, sx, sy, s_vals)
    
    # 6. Plot
    print("Plotting...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Ground Truth", "Prediction", "Variance"]
    fields = [truth, pred, var]
    
    for ax, field, title in zip(axes, fields, titles):
        im = ax.imshow(field, origin='lower', extent=[0, dom['x_max'], 0, dom['y_max']], cmap='viridis')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    
    axes[0].scatter(sx, sy, c='red', s=10, label='Samples')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, cfg['output']['plot_filename']))
    print("Done.")

if __name__ == "__main__":
    main()