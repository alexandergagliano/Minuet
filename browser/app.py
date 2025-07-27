from flask import Flask, render_template, jsonify, send_file
import h5py
import torch
import zuko
import numpy as np
import io
import base64
from PIL import Image
import os
import pickle

app = Flask(__name__)

# Global variables to store loaded data and model
data_file = None
flow_model = None
theta_mean = None
theta_std = None
latent_coords = None

def load_data_and_model():
    """Load the galaxy data and flow model"""
    global data_file, flow_model, theta_mean, theta_std, latent_coords
    
    # Load the data file
    data_path = "/Users/alexgagliano/Documents/Research/GalaxyAutoencoder/data/test_subset/maskeddaep5_results_sample_1753214750_maskdaep5.h5"
    data_file = h5py.File(data_path, 'r')
    
    # Load the flow model
    flow_path = "/Users/alexgagliano/Documents/Research/GalaxyAutoencoder/models/daep_conditional_flow_3param_5latent.pt"
    flow_model = zuko.flows.NSF(3, 5, transforms=10, hidden_features=[512] * 6)
    state_dict = torch.load(flow_path, map_location='cpu', weights_only=True)
    flow_model.load_state_dict(state_dict)
    flow_model.eval()
    
    # Calculate normalization parameters from the data
    theta = data_file['y_true'][:, 0, 0:3]  # Redshift, stellar mass, SFR
    theta_mean = theta.mean(axis=0)
    theta_std = theta.std(axis=0)
    
    # Load raw latent coordinates (5D)
    latent_coords = data_file['latent'][:, :, 0]  # Shape: (N, 5)
    
    print(f"Loaded {len(data_file['images'])} galaxies")
    print(f"Latent shape: {data_file['latent'].shape}")
    print(f"Images shape: {data_file['images'].shape}")
    print(f"Raw latent coordinates shape: {latent_coords.shape}")
    print(f"Latent dimensions available: 0, 1, 2, 3, 4")
    print(f"Latent coordinate ranges:")
    for i in range(5):
        print(f"  Dim {i}: [{latent_coords[:, i].min():.3f}, {latent_coords[:, i].max():.3f}]")

@app.route('/')
def index():
    """Main page with galaxy grid"""
    return render_template('index.html')

@app.route('/test')
def test():
    """Test page for debugging"""
    return render_template('test.html')

@app.route('/test-simple')
def test_simple():
    """Simple test page for debugging"""
    return render_template('test_simple.html')

@app.route('/minimal')
def minimal():
    """Minimal test page"""
    return render_template('index_minimal.html')

@app.route('/debug')
def debug():
    """Debug test page"""
    return render_template('index_debug.html')

@app.route('/api/galaxies')
def get_galaxies():
    """Get list of all galaxies with their basic info"""
    galaxies = []
    n_galaxies = len(data_file['images'])
    
    # Return basic info for all galaxies with raw latent coordinates
    for i in range(n_galaxies):
        galaxies.append({
            'id': i,
            'latent': latent_coords[i].tolist(),  # All 5 latent dimensions
            'true_params': data_file['y_true'][i, 0, 0:3].tolist(),  # Redshift, mass, SFR
            'latent_coords': latent_coords[i].tolist()  # Raw 5D latent coordinates
        })
    
    return jsonify({
        'total_galaxies': n_galaxies,
        'galaxies': galaxies,
        'latent_dimensions': 5
    })

@app.route('/api/galaxy/<int:galaxy_id>/image')
def get_galaxy_image(galaxy_id):
    """Get galaxy image as PNG file"""
    from flask import request
    
    if galaxy_id >= len(data_file['images']):
        return jsonify({'error': 'Galaxy ID out of range'}), 404
    
    # Get image type from query parameter, default to 'original'
    image_type = request.args.get('type', 'original')
    
    # Get the image based on type
    if image_type == 'recon':
        img = data_file['recon'][galaxy_id]  # Shape: (3, 72, 72)
    else:
        img = data_file['images'][galaxy_id]  # Shape: (3, 72, 72)
    
    # Convert to PIL Image format (H, W, C)
    img = np.transpose(img, (1, 2, 0))  # Shape: (72, 72, 3)
    
    # Clip to reasonable range and normalize
    img = np.clip(img, 0, 1)  # Ensure values are between 0 and 1
    img = (img * 255).astype(np.uint8)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return send_file(buffer, mimetype='image/png')

@app.route('/api/galaxy/<int:galaxy_id>/predictions')
def get_galaxy_predictions(galaxy_id):
    """Get posterior predictions for a galaxy"""
    if galaxy_id >= len(data_file['latent']):
        return jsonify({'error': 'Galaxy ID out of range'}), 404
    
    # Get the latent representation
    latent = data_file['latent'][galaxy_id, :, 0]  # Shape: (5,)
    latent_tensor = torch.tensor(latent, dtype=torch.float32).unsqueeze(0)
    
    # Generate samples from the flow
    with torch.no_grad():
        flow_dist = flow_model(latent_tensor)
        samples = flow_dist.sample((1000,))  # Sample 1000 times for smooth KDE
        
        # Denormalize all samples
        samples_denorm = samples.squeeze().numpy() * theta_std + theta_mean
        
        # Calculate statistics for reference
        mean_pred = samples_denorm.mean(axis=0)
        std_pred = samples_denorm.std(axis=0)
    
    # Get true values
    true_params = data_file['y_true'][galaxy_id, 0, 0:3]
    
    return jsonify({
        'galaxy_id': galaxy_id,
        'predictions': {
            'redshift': {
                'samples': samples_denorm[:, 0].tolist(),
                'mean': float(mean_pred[0]),
                'std': float(std_pred[0]),
                'true': float(true_params[0])
            },
            'stellar_mass': {
                'samples': samples_denorm[:, 1].tolist(),
                'mean': float(mean_pred[1]),
                'std': float(std_pred[1]),
                'true': float(true_params[1])
            }
        },
        'latent': latent.tolist()
    })

if __name__ == '__main__':
    print("Loading data and model...")
    load_data_and_model()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5002) 
