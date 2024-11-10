'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
from comet_ml import Experiment
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import sdf_meshing
import configargparse
import numpy as np


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=16384)
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mode', type=str, default='mlp',
               help='Options are "mlp" or "nerf"')
p.add_argument('--resolution', type=int, default=1600)

opt = p.parse_args()


class SDFDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the model.
        if opt.mode == 'mlp':
            self.model = modules.SingleBVPNet(type=opt.model_type, final_layer_factor=1, in_features=3)
        elif opt.mode == 'nerf':
            self.model = modules.SingleBVPNet(type='relu', mode='nerf', final_layer_factor=1, in_features=3)
        self.model.load_state_dict(torch.load(opt.checkpoint_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)['model_out']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

sdf_decoder = SDFDecoder()
sdf_decoder.to(DEVICE)
sdf_decoder.eval()


# def find_temperature(decoder, x, y, z_min, z_max, epsilon=1e-4):
#     device = next(decoder.parameters()).device

#     best_z = None
#     best_sdf_abs = float('inf')
#     iterations = 0
#     max_iterations = 50

#     while z_max - z_min > epsilon and iterations < max_iterations:
#         z_mid = (z_min + z_max) / 2
#         point = torch.tensor([[x, y, z_mid]], device=device)

#         with torch.no_grad():
#             sdf_value = decoder(point).item()

#         if abs(sdf_value) < best_sdf_abs:
#             best_z = z_mid
#             best_sdf_abs = abs(sdf_value)

#         if abs(sdf_value) < epsilon:
#             return z_mid

#         if sdf_value > 0:
#             z_max = z_mid
#         else:
#             z_min = z_mid
        
#         iterations += 1

#     return best_z


def find_temperature_batch(decoder, x, y, z_min, z_max, num_samples=1000):
    z_values = torch.linspace(z_min, z_max, num_samples, device=DEVICE)


    x_values = torch.full((num_samples,), x, device=DEVICE)
    y_values = torch.full((num_samples,), y, device=DEVICE)
    points = torch.stack([x_values, y_values, z_values], dim=1)

    with torch.no_grad():
        sdf_values = decoder(points).squeeze()


    abs_sdf_values = torch.abs(sdf_values)
    min_idx = torch.argmin(abs_sdf_values)
    best_z = z_values[min_idx].item()
    return best_z





experiment = Experiment(
api_key="RVIjdz27W32Dx6WrR51bEWg24",
project_name="siren",
workspace="ketiovv"
)




differences = []
squared_errors = []
total_error = 0
coords = []

# Load coordinates from the file
with open('/home/likewise-open/ADM/242575/Desktop/testing/testing_data.txt', 'r') as f:
    for line in f:
        x, y, z = map(float, line.strip().split()[:3])  # Read only the first 3 columns
        coords.append([x, y, z])

coords_tensor = torch.tensor(coords, device=DEVICE)

# Process all coordinates in batches
batch_size = 128  # Adjust batch size based on memory usage and GPU capacity
results = []
differences = []
squared_errors = []

for i in range(0, len(coords_tensor), batch_size):
    batch = coords_tensor[i:i + batch_size]
    batch_results = []
    
    # Process each (x, y) pair in the batch and find best z
    for row in batch:
        x, y, true_z = row[0].item(), row[1].item(), row[2].item()
        predicted_z = find_temperature_batch(sdf_decoder, x, y, -11, 34)
        
        difference = predicted_z - true_z
        squared_error = difference ** 2
        batch_results.append([x, y, true_z, predicted_z])
        differences.append(difference)
        squared_errors.append(squared_error)
    
    results.extend(batch_results)

    if i % 10 == 0:
        print(f"row: {i}")

# Calculate final results
total_error = sum(squared_errors)
mse = total_error / len(squared_errors) 
rmse = mse ** 0.5  
mae = sum(abs(d) for d in differences) / len(differences)  # Mean Absolute Error

# Print metrics
print(f"Total error: {total_error}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")




results = np.array(results)
with open(f'/home/likewise-open/ADM/242575/Desktop/testing_output_12000batch.txt', 'w') as f:
    for result in results:
        f.write(f"{result[0]} {result[1]} {result[2]} {result[3]}\n")
print("Results saved")

# with open(f'/home/likewise-open/ADM/242575/Desktop/{opt.experiment_name}.txt', 'w') as f:
#     for result in results:
#         f.write(f"{result[0]} {result[1]} {result[2]} {result[3]}\n")
# print("Results saved")



