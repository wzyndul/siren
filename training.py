'''Implements a generic training loop.
'''
from comet_ml import Experiment
import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil


# def find_temperature(decoder, x, y, z_min, z_max, epsilon=1e-3):
#     device = next(decoder.parameters()).device
#     # wygenerowac tensror z n wartosciami miedzy z min a z max - 
#     # moduł, minimimalny indeks, i pobieram ta wartosc dla ktorej sdf był nablizszy zero
#     while z_max - z_min > epsilon:
#         z_mid = (z_min + z_max) / 2
#         point = torch.tensor([[x, y, z_mid]], device=device)

#         with torch.no_grad():
#             sdf_value = decoder(point).item()

#         if abs(sdf_value) < epsilon:
#             return z_mid

#         if sdf_value > 0:
#             z_max = z_mid
#         else:
#             z_min = z_mid

#     return (z_min + z_max) / 2


def find_temperature(decoder, x, y, z_min, z_max, num_samples=1000):
    device = 'cuda'
    z_values = torch.linspace(z_min, z_max, num_samples, device=device)

    x_values = torch.full((num_samples,), x, device=device)
    y_values = torch.full((num_samples,), y, device=device)
    points = torch.stack([x_values, y_values, z_values], dim=1)

    model_input = {'coords': points}

    with torch.no_grad():
        output = decoder(model_input)

        sdf_values = output['model_out'].squeeze() 

    abs_sdf_values = torch.abs(sdf_values)
    min_idx = torch.argmin(abs_sdf_values)
    best_z = z_values[min_idx].item()
    return best_z






def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None):


    experiment = Experiment(
    api_key="RVIjdz27W32Dx6WrR51bEWg24",
    project_name="siren",
    workspace="ketiovv"
    )
    
    experiment.log_parameters({
        "learning_rate": lr,
        "epochs": epochs,
    })


    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    # copy settings from Raissi et al. (2019) and here 
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            # if not epoch % epochs_til_checkpoint and epoch:
            #     torch.save(model.state_dict(),
            #                os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))

            #     np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
            #                np.array(train_losses))

            epoch_loss = 0.0
            num_batches = 0
            predicted_mse = []


            correct_on_surface = 0
            total_on_surface = 0

            correct_of_surface = 0
            total_of_surface = 0

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean() 
                        train_loss.backward()
                        return train_loss
                    optim.step(closure)

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                pred_sdf = model_output['model_out']
                gt_sdf = gt['sdf']

                surface_threshold = 0.1
                of_surface_threshold = 0.9

                on_surface_mask = (gt_sdf == 0)
                pred_on_surface = (torch.abs(pred_sdf[on_surface_mask]) < surface_threshold)
                correct_on_surface += pred_on_surface.sum().item()
                total_on_surface += on_surface_mask.sum().item()

                of_surface_mask = (gt_sdf == -1)
                pred_of_surface = (torch.abs(pred_sdf[of_surface_mask]) > of_surface_threshold)
                correct_of_surface += pred_of_surface.sum().item()
                total_of_surface += of_surface_mask.sum().item()
                
                if epoch % 200 == 0:
                    step_mse = 0.0
                    for i in range(model_input['coords'].shape[1]):
                        single_point = model_input['coords'][0, :, :]
                        z_predicted = find_temperature(model, single_point[i][0].item(), single_point[i][1].item(), -11, 34)
                        z_actual = single_point[i][2]
                        mse = (z_actual - z_predicted) ** 2
                        step_mse += mse
                    
                    avg_step_mse = step_mse / model_input['coords'].shape[1]
                    predicted_mse.append(avg_step_mse.item())



                        

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_step)

                    # writer.add_scalar(loss_name, single_loss, total_steps)


                    train_loss += single_loss

                train_losses.append(train_loss.item())
                # writer.add_scalar("total_train_loss", train_loss, total_steps) TODO TO działało wczesniej ale chciałem sprawdzic
                # experiment.log_metric("total_step_loss", train_loss, step=total_steps) # na razie bede logował tylko epokami


                epoch_loss += train_loss.item()
                num_batches += 1

                # if not total_steps % steps_til_summary:
                #     torch.save(model.state_dict(),
                #                os.path.join(checkpoints_dir, 'model_current.pth'))
                #     summary_fn(model, model_input, gt, model_output, writer, total_steps)


                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                total_steps += 1




            if val_dataloader is not None: #and epoch % 200 == 0:
                print("Running validation set...")
                model.eval()
                # with torch.no_grad():
                predicted_mse_losses = []
                val_losses = []

                val_correct_on_surface = 0
                val_total_on_surface = 0


                val_correct_of_surface = 0
                val_total_of_surface = 0


                for (model_input, gt) in val_dataloader:

                    model_input = {key: value.cuda() for key, value in model_input.items()}
                    gt = {key: value.cuda() for key, value in gt.items()}

                    model_output = model(model_input)
                    pred_sdf = model_output['model_out']
                    gt_sdf = gt['sdf']

                    if epoch % 200 == 0:
                        step_mse = 0.0
                        for i in range(model_input['coords'].shape[1]):
                            single_point = model_input['coords'][0, :, :]
                            z_predicted = find_temperature(model, single_point[i][0].item(), single_point[i][1].item(), -11, 34)
                            z_actual = single_point[i][2]
                            mse = (z_actual - z_predicted) ** 2
                            step_mse += mse
                    
                        avg_step_mse = step_mse / model_input['coords'].shape[1]

                        predicted_mse_losses.append(avg_step_mse.item())



                    losses = loss_fn(model_output, gt)

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()
                        train_loss += single_loss

                    
                    val_surface_threshold = 0.1
                    val_of_surface_threshold = 0.9

                    val_on_surface_mask = (gt_sdf == 0)
                    val_pred_on_surface = (torch.abs(pred_sdf[val_on_surface_mask]) < val_surface_threshold)
                    val_correct_on_surface += val_pred_on_surface.sum().item()
                    val_total_on_surface += val_on_surface_mask.sum().item()

                    val_of_surface_mask = (gt_sdf == -1)
                    val_pred_of_surface = (torch.abs(pred_sdf[val_of_surface_mask]) > val_of_surface_threshold)
                    val_correct_of_surface += val_pred_of_surface.sum().item()
                    val_total_of_surface += val_of_surface_mask.sum().item()


                    val_losses.append(train_loss.item())

                

                val_on_surface_accuracy = val_correct_on_surface / val_total_on_surface if val_total_on_surface > 0 else 0
                val_of_surface_accuracy = val_correct_of_surface / val_total_of_surface if val_total_of_surface > 0 else 0

                experiment.log_metric("validation_on_surface_accuracy", val_on_surface_accuracy, epoch=epoch)
                experiment.log_metric("validation_of_surface_accuracy", val_of_surface_accuracy, epoch=epoch)

                epoch_avg_val_loss = np.mean(val_losses)
                experiment.log_metric("validation_avg_loss", epoch_avg_val_loss, epoch=epoch)

                if epoch % 200 == 0:
                    epoch_avg_predicted_loss_mse = np.mean(predicted_mse_losses)
                    experiment.log_metric("validation__predicted_z_mse", epoch_avg_predicted_loss_mse, epoch=epoch)


                        

                model.train()

            if epoch % 200 == 0:
                predicted_mse = np.mean(predicted_mse)
                experiment.log_metric("training_predicted_z_mse", predicted_mse, epoch=epoch)

            on_surface_accuracy = correct_on_surface / total_on_surface if total_on_surface > 0 else 0
            of_surface_accuracy = correct_of_surface / total_of_surface if total_of_surface > 0 else 0

            experiment.log_metric("training_on_surface_accuracy", on_surface_accuracy, epoch=epoch)
            experiment.log_metric("training_of_surface_accuracy", of_surface_accuracy, epoch=epoch)


            avg_epoch_loss = epoch_loss / num_batches
            experiment.log_metric("epoch", epoch)
            experiment.log_metric("training_avg_loss", avg_epoch_loss, epoch=epoch)

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))




class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)