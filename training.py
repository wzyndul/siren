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
torch.manual_seed(0)
np.random.seed(0)


def find_temperature(decoder, x, y, z_min, z_max, num_samples=1000):
    device = 'cuda'
    # breakpoint()
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


def inverse_transform(value, coord_min, coord_max):
    value /= 2.0
    value += 0.5
    value = value * (coord_max - coord_min) + coord_min
    return value



def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None):


    experiment = Experiment(
    api_key="",
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
            epoch_losses = {}
            num_batches = 0
            predicted_mse = []


            correct_on_surface = 0
            total_on_surface = 0

            correct_of_surface = 0
            total_of_surface = 0
            # import pdb;pdb.set_trace()
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

                surface_threshold = 0.05


                on_surface_mask = (gt_sdf == 0)
                pred_on_surface = (torch.abs(pred_sdf[on_surface_mask]) < surface_threshold)
                correct_on_surface += pred_on_surface.sum().item()
                total_on_surface += on_surface_mask.sum().item()

                of_surface_mask = (gt_sdf == -1)
                pred_of_surface = (torch.abs(pred_sdf[of_surface_mask]) > surface_threshold)
                correct_of_surface += pred_of_surface.sum().item()
                total_of_surface += of_surface_mask.sum().item()
                
                if epoch % 200 == 0 and epoch != 0:
                    step_mse = 0.0
                    min_max_mean = train_dataloader.dataset.get_min_max_mean() # zwraca tylko min max
                    # print(f"min max mean{min_max_mean}")
                    for i in range(model_input['coords'].shape[1]):
                        all_sdf_gt = gt_sdf_all = gt['sdf'][0, :, :]
                        all_points = model_input['coords'][0, :, :]
                        
                        if all_sdf_gt[i][0].item() == 0:
                            # print(f"coords: {all_points[i][0].item()}, {all_points[i][1].item()}, {all_points[i][2].item()}")

                            z_predicted = find_temperature(model, all_points[i][0].item(), all_points[i][1].item(), -1, 1)

                            # print(f"z predicted: {z_predicted}")

                            z_predicted = inverse_transform(z_predicted, min_max_mean[0], min_max_mean[1])

                            # print(f"z predicted after inverse transorm: {z_predicted}")

                            z_actual = inverse_transform(all_points[i][2].item(), min_max_mean[0], min_max_mean[1])

                            # print(f"z true after inverse transorm: {z_actual}")

                            mse = (z_actual - z_predicted) ** 2

                            # print(f"mse: {mse}")
                            step_mse += mse.item()
                    
                    # print(f"step mse: {step_mse}")
                    # print(f"size: {model_input['coords'].shape[1]}")
                    avg_step_mse = step_mse / (model_input['coords'].shape[1]/2)

                    # print(f"avg_step_mse: {avg_step_mse}")
                    predicted_mse.append(avg_step_mse)



                        

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_step)

                    # writer.add_scalar(loss_name, single_loss, total_steps)
                    epoch_losses[loss_name] = epoch_losses.get(loss_name, 0) + single_loss.item()

                    train_loss += single_loss

                train_losses.append(train_loss.item()) #holds step losses for whole training proccess
                # writer.add_scalar("total_train_loss", train_loss, total_steps)



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

                epoch_losses_val = {}

                # epoch_losses_val_surface = {}
                # epoch_losses_val_of_surface = {}
                step_val_nr = 0
                for step, (model_input, gt) in enumerate(val_dataloader):

                    model_input = {key: value.cuda() for key, value in model_input.items()}
                    gt = {key: value.cuda() for key, value in gt.items()}

                    model_output = model(model_input)
                    pred_sdf = model_output['model_out']
                    gt_sdf = gt['sdf']

                    if epoch % 200 == 0 and epoch != 0:
                        step_mse = 0.0
                        min_max_mean = val_dataloader.dataset.get_min_max_mean()  #zwraca tylko min amx

                        for i in range(model_input['coords'].shape[1]):
                            all_sdf_gt = gt_sdf_all = gt['sdf'][0, :, :]
                            all_points_val = model_input['coords'][0, :, :]

                            if all_sdf_gt[i][0].item() == 0:

                                z_predicted = find_temperature(model, all_points_val[i][0].item(), all_points_val[i][1].item(), -1, 1)


                                z_actual = inverse_transform(all_points_val[i][2].item(), min_max_mean[0], min_max_mean[1])

                                z_predicted = inverse_transform(z_predicted, min_max_mean[0], min_max_mean[1])


                                mse = (z_actual - z_predicted) ** 2

                               

                                step_mse += mse.item()

                        

                        avg_step_mse = step_mse / (model_input['coords'].shape[1]/2)

                        predicted_mse_losses.append(avg_step_mse)



                    losses = loss_fn(model_output, gt)


                    # gt_surface = {}
                    # gt_of_surface = {}

                    # # Process the ground truth dictionary
                    # for key, tensor in gt.items():
                    #     gt_surface[key] = tensor[:, :tensor.size(1) // 2]  # Keep the first half
                    #     gt_of_surface[key] = tensor[:, tensor.size(1) // 2:]  # Keep the second half

                    # model_output_surface = {}
                    # model_output_of_surface = {}

                    # # Process the model output dictionary
                    # for key, tensor in model_output.items():
                    #     model_output_surface[key] = tensor[:, :tensor.size(1) // 2]  # Keep the first half
                    #     model_output_of_surface[key] = tensor[:, tensor.size(1) // 2:]  # Keep the second half

                    # # Print results
                    # print("Model output")
                    # print(model_output)

                    # print("Model output surface (first half only)")
                    # print(model_output_surface)

                    # print("Model output of surface (second half only)")
                    # print(model_output_of_surface)



                    # losses_surface = loss_fn(model_output_surface, gt_surface)
                    # losses_of_surface = loss_fn(model_output_of_surface, gt_of_surface)

                    # for loss_name, loss in losses_surface.items():
                    #     single_loss = loss.mean()
                    #     epoch_losses_val_surface[loss_name] = epoch_losses_val_surface.get(loss_name, 0) + single_loss.item()

                    
                    # for loss_name, loss in losses_of_surface.items():
                    #     single_loss = loss.mean()
                    #     epoch_losses_val_of_surface[loss_name] = epoch_losses_val_of_surface.get(loss_name, 0) + single_loss.item()
                    
                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()
                        train_loss += single_loss
                        epoch_losses_val[loss_name] = epoch_losses_val.get(loss_name, 0) + single_loss.item()

                    
                    
                    val_losses.append(train_loss.item())
                    val_surface_threshold = 0.05


                    val_on_surface_mask = (gt_sdf == 0)
                    val_pred_on_surface = (torch.abs(pred_sdf[val_on_surface_mask]) < val_surface_threshold)
                    val_correct_on_surface += val_pred_on_surface.sum().item()
                    val_total_on_surface += val_on_surface_mask.sum().item()

                    val_of_surface_mask = (gt_sdf == -1)
                    val_pred_of_surface = (torch.abs(pred_sdf[val_of_surface_mask]) > val_surface_threshold)
                    val_correct_of_surface += val_pred_of_surface.sum().item()
                    val_total_of_surface += val_of_surface_mask.sum().item()

                    step_val_nr += 1


                    

                

                val_on_surface_accuracy = val_correct_on_surface / val_total_on_surface if val_total_on_surface > 0 else 0
                val_of_surface_accuracy = val_correct_of_surface / val_total_of_surface if val_total_of_surface > 0 else 0

                experiment.log_metric("validation_on_surface_accuracy", val_on_surface_accuracy, epoch=epoch)
                experiment.log_metric("validation_of_surface_accuracy", val_of_surface_accuracy, epoch=epoch)

                epoch_avg_val_loss = np.mean(val_losses)
                experiment.log_metric("validation_avg_loss", epoch_avg_val_loss, epoch=epoch)

                for loss_name, total_loss in epoch_losses_val.items(): #TODO zawsze ustalam to tak, że bedzie 1 batch wiec zawsze powinno byc git
                    experiment.log_metric(f"validation_{loss_name}", total_loss / step_val_nr, epoch=epoch) # tak to trzeba podzielic przez liczbe stepów

                # combined_losses = {}

               
                # loss_surface = 0.0 
                # for loss_name, total_loss in epoch_losses_val_surface.items():
                #     loss_surface += total_loss
                #     combined_losses[loss_name] = combined_losses.get(loss_name, 0) + total_loss
                #     experiment.log_metric(f"validation_{loss_name}_surface", total_loss, epoch=epoch)

                # experiment.log_metric(f"validation_avg_total_loss_surface", loss_surface, epoch=epoch)

                # loss_of_surface = 0.0
                # for loss_name, total_loss in epoch_losses_val_of_surface.items():
                #     loss_of_surface += total_loss
                #     combined_losses[loss_name] = combined_losses.get(loss_name, 0) + total_loss
                #     experiment.log_metric(f"validation_{loss_name}_of_surface", total_loss, epoch=epoch)

                # experiment.log_metric(f"validation_avg_total_loss_of_surface", loss_of_surface, epoch=epoch)
                
                # experiment.log_metric(f"validation_avg_total_loss_of_surface", loss_of_surface, epoch=epoch)

                # # Log each combined loss
                # for loss_name, total_loss in combined_losses.items():
                #     experiment.log_metric(f"validation_{loss_name}_both_surface_and_of_surface", total_loss, epoch=epoch)

  

                if epoch % 200 == 0 and epoch != 0:
                    epoch_avg_predicted_loss_mse = np.mean(predicted_mse_losses)
                    experiment.log_metric("validation_predicted_z_mse", epoch_avg_predicted_loss_mse, epoch=epoch)


                        

                model.train()

            if epoch % 200 == 0 and epoch != 0:
                predicted_mse = np.mean(predicted_mse)
                experiment.log_metric("training_predicted_z_mse", predicted_mse, epoch=epoch)

            # on_surface_accuracy = correct_on_surface / total_on_surface if total_on_surface > 0 else 0
            # of_surface_accuracy = correct_of_surface / total_of_surface if total_of_surface > 0 else 0

            # experiment.log_metric("training_on_surface_accuracy", on_surface_accuracy, epoch=epoch)
            # experiment.log_metric("training_of_surface_accuracy", of_surface_accuracy, epoch=epoch)


            avg_epoch_loss = epoch_loss / num_batches
            experiment.log_metric("epoch", epoch)
            experiment.log_metric("training_avg_loss", avg_epoch_loss, epoch=epoch)

            # for loss_name, total_loss in epoch_losses.items():
            #     avg_loss = total_loss / num_batches
            #     experiment.log_metric(f"training_{loss_name}", avg_loss, epoch=epoch)



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
