import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import the toolkit from the src folder
# This handles the path if you run this script from 'tests/' or root
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from geostat_toolkit import generate_synthetic_field, run_kriging, save_to_vtk

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # 1. Load Configuration
    config_path = os.path.join(current_dir, 'demo_config.yaml')
    print(f"Loading config from: {config_path}")
    cfg = load_config(config_path)

    # 2. Generate Synthetic "Ground Truth"
    print("Generating synthetic field...")
    sim_cfg = cfg['simulation']
    dom_cfg = cfg['domain']
    
    true_x, true_y, true_field = generate_synthetic_field(
        x_max=dom_cfg['x_max'],
        y_max=dom_cfg['y_max'],
        nx=dom_cfg['nx'],
        ny=dom_cfg['ny'],
        model_type=sim_cfg['model_type'],
        variance=sim_cfg['variance'],
        length_scale=sim_cfg['length_scale'],
        seed=sim_cfg['seed']
    )

    # 3. Sampling (Create artificial observations)
    print("Sampling observation points...")
    n_samples = cfg['sampling']['n_samples']
    rng = np.random.default_rng(cfg['sampling']['seed'])
    
    # Random indices
    idx_x = rng.integers(0, dom_cfg['nx'], n_samples)
    idx_y = rng.integers(0, dom_cfg['ny'], n_samples)
    
    sample_x = true_x[idx_x]
    sample_y = true_y[idx_y]
    sample_values = true_field[idx_y, idx_x] # Note: indexing often (y, x) for grids

    # 4. Run Kriging
    # Check if user wants auto-optimization or manual override
    user_model = cfg['kriging'].get('variogram_model')
    
    params = None
    if user_model:
        # Construct manual params dict
        params = {
            "model": user_model, 
            "angle": cfg['kriging'].get('anisotropy_angle', 0.0)
        }
        print(f"Using manual Kriging parameters: {params}")
    else:
        print("Auto-optimizing variogram model...")

    pred_field, var_field, final_params = run_kriging(
        sample_x, sample_y, sample_values,
        grid_x=true_x, grid_y=true_y,
        params=params
    )
    
    print(f"Kriging Complete using model: {final_params['model']} (Angle: {final_params['angle']:.1f})")

    # 5. Visualization (Matplotlib)
    print("Generating comparison plot...")
    out_folder = cfg['output']['folder']
    os.makedirs(out_folder, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Ground Truth
    im0 = axes[0].imshow(true_field, origin='lower', extent=[0, dom_cfg['x_max'], 0, dom_cfg['y_max']], cmap='viridis')
    axes[0].set_title("Ground Truth")
    axes[0].scatter(sample_x, sample_y, c='red', s=5, label='Samples')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot Prediction
    im1 = axes[1].imshow(pred_field, origin='lower', extent=[0, dom_cfg['x_max'], 0, dom_cfg['y_max']], cmap='viridis')
    axes[1].set_title("Kriging Prediction")
    plt.colorbar(im1, ax=axes[1])
    
    # Plot Variance (Uncertainty)
    im2 = axes[2].imshow(var_field, origin='lower', extent=[0, dom_cfg['x_max'], 0, dom_cfg['y_max']], cmap='plasma')
    axes[2].set_title("Kriging Variance")
    plt.colorbar(im2, ax=axes[2])
    
    plot_path = os.path.join(out_folder, cfg['output']['plot_filename'])
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

    # 6. Export to ParaView
    print("Exporting VTK file...")
    vtk_path = os.path.join(out_folder, cfg['output']['vtk_filename'])
    
    # We export a dict of fields
    data_to_export = {
        "Prediction": pred_field,
        "Variance": var_field,
        "GroundTruth": true_field
    }
    
    save_to_vtk(vtk_path, data_to_export, true_x, true_y)
    print(f"VTK saved to: {vtk_path}.vtr")

if __name__ == "__main__":
    main()