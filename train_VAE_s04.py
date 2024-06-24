from utils.logger import logger
import torch.nn.parallel
import torch.nn as nn
import torch.optim
import torch
from utils.loaders import ActionNetDataset, EpicKitchensDataset
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import wandb
import matplotlib.pyplot as plt
from  sklearn.manifold import TSNE
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# global variables among training functions
training_iterations = 0
modalities = None
np.random.seed(13696641)
torch.manual_seed(13696641)

# with this script we trained and tested FC_VAE.VariationalAutoencoder to reconstruct features from the EMG modality
def init_operations():
    """
    parse all the arguments, generate the logger, check gpus to be used and wandb
    """
    logger.info("Running with parameters: " + pformat_dict(args, indent=1))

    # this is needed for multi-GPUs systems where you just want to use a predefined set of GPUs
    if args.gpus is not None:
        logger.debug('Using only these GPUs: {}'.format(args.gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
    
    if args.wandb_name is not None:
        WANDB_KEY = "c87fa53083814af2a9d0ed46e5a562b9a5f8b3ec" # Salvatore's key
        if os.getenv('WANDB_KEY') is not None:
            WANDB_KEY = os.environ['WANDB_KEY']
            logger.info("Using key retrieved from enviroment.")
        wandb.login(key=WANDB_KEY)
        run = wandb.init(project="FC-VAE(rgb)", entity="egovision-aml22", name = "VAE_RGB_EMG")
        wandb.run.name = "VAE_RGB_EMG"

def main():
    global training_iterations, modalities
    init_operations()
    modalities = args.modality
    # device where everything is run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # these dictionaries are for more multi-modal training/testing, each key is a modality used
    model = getattr(model_list, args.models.VAE.model)(args.train.RGB.feature_size, 
                                                              args.train.bottleneck_size, 
                                                              args.train.EMG.feature_size,
                                                              resume_from=args.last_model)
    
    if args.action == "train":
        model.load_last_model()
        model.load_on(device)

        train_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'train', args.dataset, None, None, None,
                                                                       None, load_feat=True),
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                     'test', args.dataset, None, None, None,
                                                                     None, load_feat=True),
                                                 batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        
        loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                       'train' , args.dataset, None, None, None,
                                                                       None, load_feat=True, additional_info=True),
                                                   batch_size=1, shuffle=False,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
        
        loader_test = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                       "test", args.dataset, None, None, None,
                                                                       None, load_feat=True, additional_info=True),
                                                   batch_size=1, shuffle=False,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
        
        ae = train(model, train_loader, val_loader, device, args.models.VAE)
        save_model(ae, "VAE_RGB_EMG")
        logger.info(f"Model saved")
        logger.info(f"TRAINING VAE FINISHED, RECONSTUCTING FEATURES...")

        reconstructed_features, results = reconstruct_simulated(model, loader, device, "train", filename="S04", debug = True)
        logger.debug(f"Results on train: {results}")
        reconstructed_features = reconstruct_simulated(model, loader_test, device, "test", filename="S04")
    elif args.action == "reconstruct":
        model.load_last_model()
        model.load_on(device)

        loader = torch.utils.data.DataLoader(EpicKitchensDataset("D1", ['RGB'],
                                                                       'train', args.dataset,  None, args.train.num_clips, None,
                                                                       load_feat=True, additional_info=True),
                                                   batch_size=1, shuffle=False,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        
        loader_test = torch.utils.data.DataLoader(EpicKitchensDataset("D1", ['RGB'],
                                                                       'test', args.dataset,  None, args.train.num_clips, None,
                                                                       load_feat=True, additional_info=True),
                                                   batch_size=1, shuffle=False,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        logger.info(f"Generating FEATURES...")
        reconstructed_features = reconstruct(model, loader, device, "train", filename="Epic")
        reconstructed_features = reconstruct(model, loader_test, device, "test", filename="Epic")

    else:
        raise NotImplementedError(f"Action {args.action} not implemented")

def validate(autoencoder, val_dataloader, device, reconstruction_loss):
    total_loss = 0
    autoencoder.train(False)

    for _, (data, _) in enumerate(val_dataloader):
        RGB_clip = data['RGB'].to(device)
        EMG_clip = data['EMG'].permute(1, 0, 2).squeeze(0).to(device)

        x_hat, _, _, _ = autoencoder(RGB_clip)
        total_loss += reconstruction_loss(x_hat, EMG_clip)
    return total_loss/(5 * len(val_dataloader))

def train(autoencoder, train_dataloader, val_dataloader, device, model_args):
    logger.info(f"Start VAE training.")

    opt = build_optimizer(autoencoder, "adam", model_args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=model_args.lr_steps, gamma=model_args.lr_gamma)

    reconstruction_loss = nn.MSELoss(reduction='mean')

    autoencoder.train(True)
    beta = np.ones(model_args.epochs) * model_args.beta
    
    for epoch in range(model_args.epochs):
        # train_loop
        total_loss = 0 # total loss for the epoch
        for _, (data, _) in enumerate(train_dataloader):
            opt.zero_grad()
            
            RGB_clip = data['RGB'].to(device)
            EMG_clip = data['EMG'].permute(1, 0, 2).squeeze(0).to(device)

            
            x_hat, _, mean, log_var = autoencoder(RGB_clip)

            mse_loss = reconstruction_loss(x_hat, EMG_clip)                              #  compute the reconstruction loss
            kld_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())  #  compute the KLD loss
            loss = mse_loss + beta[epoch] * kld_loss
            # generate an error if loss is nan
            if loss.isnan():
                raise ValueError("Loss is NaN.")
            loss.backward()
            opt.step()
            wandb.log({"Beta": beta[epoch], "MSE LOSS": mse_loss, 'KLD_loss': kld_loss, 'loss': loss, 'lr': scheduler.get_last_lr()[0]})
            total_loss += loss.item()
        if epoch % 10 == 0:
            wandb.log({"validation_loss": validate(autoencoder, val_dataloader, device, reconstruction_loss)})
        print(f"[{epoch+1}/{model_args.epochs}] - Total loss: {total_loss}")
        scheduler.step()
    return autoencoder

#Simulated because we are using a test set, so we also want to compute the error
def reconstruct_simulated(autoencoder, dataloader, device, split=None, filename='test',**kwargs):
    """
    Reconstruct the features using the trained autoencoder
    - autoencoder: the trained autoencoder
    - dataloader: the dataloader to use
    - device: the device to use
    - split: the split to use
    - kwargs: additional arguments
    """
    debug = kwargs.get('debug', False)
    result = {'features': []}
    # for debugging purpose, I introduce also a loss in reconstruction
    reconstruction_loss = nn.MSELoss()
    avg_video_level_loss = 0
    autoencoder.train(False)
    with torch.no_grad():
        for _, (data, label, video_name, uid) in enumerate(dataloader):

            RGB_clip = data['RGB'].to(device)
            EMG_clip = data['EMG'].permute(1, 0, 2).squeeze(0).to(device)
            x_hat, _, _, _ = autoencoder(RGB_clip)  
            
            avg_video_level_loss += reconstruction_loss(EMG_clip, x_hat)

            x_hat = x_hat.cpu()
            
            result['features'].append({
                'features_EMG': x_hat.numpy(), 
                'label': label.item(), 
                'uid': uid.item(), 
                'video_name': video_name
            })
    try:
        with open(os.path.join(f'./saved_features/reconstructed_EMG_{filename}', f"{'ActionNet'}_{split}.pkl"), "wb") as file:
            pickle.dump(result, file)
        logger.info(f"Saved {'ActionNet'}_{split}.pkl")
    except Exception as e:
        logger.warning(f"Error while saving the file: {e}")
    
    if debug:
        return result, {'total_loss': avg_video_level_loss, 'avg_loss': avg_video_level_loss/len(dataloader)}
    else:
        return result
    

def reconstruct(autoencoder, dataloader, device, split=None, filename='test'):
    """
    Reconstruct the features using the trained autoencoder
    - autoencoder: the trained autoencoder
    - dataloader: the dataloader to use
    - device: the device to use
    - split: the split to use
    - kwargs: additional arguments
    """
    result = {'features': []}

    autoencoder.train(False)
    with torch.no_grad():
        for _, (data, label, video_name, uid) in enumerate(dataloader):

            RGB_data = data['RGB'].permute(1, 0, 2).to(device)
            EMG_clips = [] #we will append here the new clips

            for i in range(args.save.num_clips):
                RGB_clip = RGB_data[i]
                
                x_hat, _, _, _ = autoencoder(RGB_clip)  

                x_hat = x_hat.cpu()
                EMG_clips.append(x_hat)
            
            #need to change the EMG clip to align it with the dimensionality of the RGB tensor
            result['features'].append({
                'features_EMG': torch.stack(EMG_clips, dim = 0).permute(1, 0, 2).squeeze(0).numpy(),
                'label': label.item(), 
                'uid': uid.item(), 
                'video_name': video_name
            })
    try:
        with open(os.path.join(f'./extracted_features', f"{'ActionNet'}_D1_{split}.pkl"), "wb") as file:
            pickle.dump(result, file)
        logger.info(f"Saved {'ActionNet'}_{split}.pkl")
    except Exception as e:
        logger.warning(f"Error while saving the file: {e}")
    
    return result


def save_model(model, filename):
    try:
        torch.save({'encoder': model.encoder.state_dict(), 'decoder': model.decoder.state_dict()}, 
                   os.path.join('./saved_models/VAE_RGB_EMG', filename))
    except Exception as e:
        logger.info("An error occurred while saving the checkpoint:")
        logger.info(e)

def load_model(ae, path):
    state_dict = torch.load(path)["model_state_dict"]
    ae.load_state_dict(state_dict, strict=False)

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer

if __name__ == '__main__':
    main()