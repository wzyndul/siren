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
import copy

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')

# General training options
p.add_argument('--batch_size', type=int, default=16384)
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mode', type=str, default='mlp',
               help='Options are "mlp" or "nerf"')
p.add_argument('--resolution', type=int, default=1600)
p.add_argument('--point_cloud_path', type=str)
p.add_argument('--save_location', type=str)

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



def find_temperature(decoder, x, y, z_min, z_max, num_samples=1000):
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
api_key="",
project_name="siren",
workspace="ketiovv"
)




def inverse_transform(value, coord_min, coord_max):
    value /= 2.0
    value += 0.5
    value = value * (coord_max - coord_min) + coord_min
    return value
    





differences = []
squared_errors = []
total_error = 0
coords = []

with open(opt.point_cloud_path, 'r') as f:
    for line in f:
        x, y, z = map(float, line.strip().split()[:3])
        coords.append([x, y, z])

original_coords = copy.deepcopy(coords)
coords = np.array(coords)




# coord_max = np.amax(coords)
# coord_min = np.amin(coords)

coord_max = np.amax(coords, axis=0, keepdims=True)
coord_min = np.amin(coords, axis=0, keepdims=True)

coord_max = np.array([[35.0, 71.0, 28.101281738281273]])
coord_min = np.array([[-24, 36, -9.947546386718727]])





coords[:, :2] = (coords[:, :2] - coord_min[0, :2]) / (coord_max[0, :2] - coord_min[0, :2])


global_min_z = -9.947546386718727
global_max_z = 28.101281738281273

# global_min_z = -12
# global_max_z = 32  

global_min_z = np.float64(global_min_z)
global_max_z = np.float64(global_max_z)


coords[:, -1] = (coords[:, -1] - global_min_z) / (global_max_z - global_min_z)


coords -= 0.5
coords *= 2.





coords_tensor = torch.tensor(coords, device=DEVICE)

results = []
differences = []
squared_errors = []


for i, row in enumerate(coords_tensor):
    x, y, true_z = row[0].item(), row[1].item(), row[2].item()

    
    predicted_z = find_temperature(sdf_decoder, x, y, -1, 1)



    x = original_coords[i][0]
    y = original_coords[i][1]

    true_z = inverse_transform(true_z, global_min_z, global_max_z)
    predicted_z = inverse_transform(predicted_z, global_min_z, global_max_z)


    difference = true_z - predicted_z
    squared_error = difference ** 2
    
    results.append([x, y, true_z, predicted_z])
    differences.append(difference)
    squared_errors.append(squared_error)

    if i % 1000 == 0:
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
with open(opt.save_location, 'w') as f:
    for result in results:
        f.write(f"{result[0]} {result[1]} {result[2]} {result[3]}\n")
print("Results saved")





