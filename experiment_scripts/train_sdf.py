'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader, random_split
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1400)
p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=5e-5') #było 1e-4
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--point_cloud_path', type=str, default='/home/sitzmann/data/point_cloud.xyz',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--validation_dataset', type=str)
p.add_argument('--on_surface_points_val', type=int)

opt = p.parse_args()


sdf_dataset = dataio.PointCloud(opt.point_cloud_path, on_surface_points=opt.batch_size, generate_points=True)

# val_dataset = dataio.PointCloud("/home/likewise-open/ADM/242575/Desktop/training_test/20k_rest.txt", on_surface_points=70147, generate_points=True)
# val_dataset = dataio.PointCloud(opt.validation_dataset, on_surface_points=opt.on_surface_points_val, generate_points=True)


train_dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
# val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3)
else:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3)
model.cuda()

# Define the loss
loss_fn = loss_functions.sdf
summary_fn = utils.write_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model, train_dataloader=train_dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, val_dataloader=None, double_precision=False,
               clip_grad=True)
