import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import the toolkit from the src folder
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from geostat_toolkit import generate_synthetic_field, run_kriging, save_to_vtk
except ImportError as e:
    print(f"Error importing geostat_toolkit: {e}")
    print(f"Make sure the package is installed or src path is correct: {src_path}")
    sys.exit(1)

def load_config(config_path):
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)

def main():
    # 1. Load Configuration
    config_path = os.path.join(current_dir, 'demo_config.yaml')
    print(f"Loading config from: {config_path}")
    cfg = load_config(config_path)

    # 2. Generate Synthetic "Ground Truth"
    print("Generating synthetic field...")
    sim_cfg = cfg['simulation']
    dom_cfg = cfg['domain']
    
    try:
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
    except Exception as e:
        print(f"Error generating synthetic field: {e}")
        return

    # 3. Sampling (Create artificial observations)
    print("Sampling observation points...")
    n_samples = cfg['sampling']['n_samples']
    rng = np.random.default_rng(cfg['sampling']['seed'])
    
    # Random indices
    idx_x = rng.integers(0, dom_cfg['nx'], n_samples)
    idx_y = rng.integers(0, dom_cfg['ny'], n_samples)
    
    sample_x = true_x[idx_x]
    sample_y = true_y[idx_y]
    # Fixed indexing: use linear indexing for 2D array
    sample_values = true_field[idx_y, idx_x]

    # 4. Run Kriging
    user_model = cfg['kriging'].get('variogram_model')
    
    params = None
    if user_model:
        params = {
            "model": user_model, 
            "angle": cfg['kriging'].get('anisotropy_angle', 0.0)
        }
        print(f"Using manual Kriging parameters: {params}")
    else:
        print("Auto-optimizing variogram model...")

    try:
        pred_field, var_field, final_params = run_kriging(
            sample_x, sample_y, sample_values,
            grid_x=true_x, grid_y=true_y,
            params=params
        )
    except Exception as e:
        print(f"Error running kriging: {e}")
        return
    
    print(f"Kriging Complete using model: {final_params['model']} (Angle: {final_params['angle']:.1f})")

    # 5. Visualization (Matplotlib)
    print("Generating comparison plot...")
    out_folder = cfg['output']['folder']
    os.makedirs(out_folder, exist_ok=True)
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot Ground Truth
        im0 = axes[0].imshow(true_field, origin='lower', 
                           extent=[0, dom_cfg['x_max'], 0, dom_cfg['y_max']], 
                           cmap='viridis')
        axes[0].set_title("Ground Truth")
        axes[0].scatter(sample_x, sample_y, c='red', s=5, label='Samples')
        axes[0].legend()
        plt.colorbar(im0, ax=axes[0])
        
        # Plot Prediction
        im1 = axes[1].imshow(pred_field, origin='lower', 
                           extent=[0, dom_cfg['x_max'], 0, dom_cfg['y_max']], 
                           cmap='viridis')
        axes[1].set_title("Kriging Prediction")
        plt.colorbar(im1, ax=axes[1])
        
        # Plot Variance (Uncertainty)
        im2 = axes[2].imshow(var_field, origin='lower', 
                           extent=[0, dom_cfg['x_max'], 0, dom_cfg['y_max']], 
                           cmap='plasma')
        axes[2].set_title("Kriging Variance")
        plt.colorbar(im2, ax=axes[2])
        
        plot_path = os.path.join(out_folder, cfg['output']['plot_filename'])
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()  # Free memory
        print(f"Plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error creating plots: {e}")

    # 6. Export to ParaView
    print("Exporting VTK file...")
    vtk_path = os.path.join(out_folder, cfg['output']['vtk_filename'])
    
    data_to_export = {
        "Prediction": pred_field,
        "Variance": var_field,
        "GroundTruth": true_field
    }
    
    try:
        save_to_vtk(vtk_path, data_to_export, true_x, true_y)
        print(f"VTK saved to: {vtk_path}.vtr")
    except Exception as e:
        print(f"Error saving VTK file: {e}")

if __name__ == "__main__":
    main()
