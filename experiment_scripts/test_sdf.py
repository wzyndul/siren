'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
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


def find_temperature(decoder, x, y, z_min, z_max, epsilon=1e-3):
    """
    Find the temperature (Z) value where the SDF is closest to zero for given X and Y coordinates.
    :param decoder: The trained neural network
    :param x: X coordinate
    :param y: Y coordinate
    :param z_min: Minimum possible temperature
    :param z_max: Maximum possible temperature
    :param epsilon: Tolerance for considering SDF close enough to zero and for search interval
    :return: Estimated temperature (Z) value closest to SDF=0
    """
    device = next(decoder.parameters()).device

    while z_max - z_min > epsilon:
        z_mid = (z_min + z_max) / 2
        point = torch.tensor([[x, y, z_mid]], device=device)

        with torch.no_grad():
            sdf_value = decoder(point).item()

        if abs(sdf_value) < epsilon:
            return z_mid

        if sdf_value > 0:
            z_max = z_mid
        else:
            z_min = z_mid

    return (z_min + z_max) / 2


sdf_decoder = SDFDecoder()


sdf_decoder.eval()

coords = []
with open('/home/likewise-open/ADM/242575/Desktop/input_cords.xyz', 'r') as f:
    for line in f:
        x, y, z = map(float, line.strip().split()[:3])  # Read only the first 3 columns
        coords.append([x, y, z])

results = []
for row in coords:
    predicted_z = find_temperature(sdf_decoder, row[0], row[1], -30, 45)
    results.append([row[0], row[1], row[2], predicted_z])

results = np.array(results)

# Save the results to "results.txt"
with open('/home/likewise-open/ADM/242575/Desktop/results2.txt', 'w') as f:
    for result in results:
        f.write(f"{result[0]} {result[1]} {result[2]} {result[3]}\n")
print("Results saved to 'results2.txt'")




# sdf_decoder.eval() # moja linijka
#
# coords = []
# with open('/home/likewise-open/ADM/242575/Desktop/input_cords.xyz', 'r') as f:
#     for line in f:
#         x, y, z = map(float, line.strip().split()[:3])  # Read only the first 3 columns
#         coords.append([x, y, z])
#
# # Convert to a PyTorch tensor and move to GPU
# coords = torch.tensor(coords, dtype=torch.float32).cuda()
#
# # Process in batches due to memory constraints
# results = []
# for i in range(0, len(coords), opt.batch_size):
#     batch_coords = coords[i:i + opt.batch_size]
#     with torch.no_grad():
#         batch_results = sdf_decoder(batch_coords)
#     results.append(batch_results.cpu().numpy())
#
# # Concatenate all results into a single array
# results = np.concatenate(results, axis=0)
#
# # Save the results to "results.txt"
# with open('/home/likewise-open/ADM/242575/Desktop/results.txt', 'w') as f:
#     for result in results:
#         f.write(f"{result[0]}\n")
#
# print("Results saved to 'results.txt'")




# original code

#
# root_path = os.path.join(opt.logging_root, opt.experiment_name)
# utils.cond_mkdir(root_path)
#
# sdf_meshing.create_mesh(sdf_decoder, os.path.join(root_path, 'test'), N=opt.resolution)
