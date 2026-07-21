#!/usr/bin/env python3

import os
import click

import numpy as np
import torch

import glob
from pathlib import Path
import json

from network import set_seed, get_class_weights
from network import load_model_only, load_model, save_model, create_model
from network import accuracy

from data import SegmentationDataset, SegmentationBunch

from tqdm.auto import tqdm

def to_tensor(arr, device):
    # If saved without .numpy(), extract the raw object first
    if arr.dtype == object:
        arr = arr.item()  # Unwraps the saved tensor/array object
    return torch.as_tensor(arr, dtype=torch.float32, device=device)

@click.command()
@click.option('-v', '--view', default='W', show_default=True,
              type=click.Choice(['W', 'U', 'V', 'UV']), help='TPC view to process')
@click.option('-c', '--classes', default=20, show_default=True,
              help='Number of semantic classes')
@click.option('-p', '--path', show_default=True,
              default='/home/msotgia/vertexOnEaf/ICARUS_DlVertex_HDF5', 
              help='Path where to find the vertex HDF5 files for the training')
@click.option('--vertex-pass', default=1, show_default=True,
              help='Vertex training pass')
@click.option('-s', '--sample', type=(str, float), multiple=True, required=True, 
              help='Samples to use and relative fraction')
@click.option('-n', '--model-name', show_default=True,
              default='icarus_fully_balanced_beam_flavour', 
              help='Name of the model that will be cached')
@click.option('-e', '--epochs', default=20, show_default=True,
              help='Number of epochs to use for the training')
@click.option('-b', '--batch-size', default=150, show_default=True,
              help='Size of the batches used for the training step')
@click.option('--validation-pct', default=0.25, show_default=True,
              help='Percentage of sample to use for validation')
@click.option('--seed', default=42, help='Seed used to randomly sample the training')
@click.option('--cache', default=None,
              help=('Whether to use cached weights/model training results.'
                    ' Requires passing the path to the cache'))
@click.option('--cache-path', default='outputs', show_default=True,
              help='Path where to save weights/losses cache and per-epoch trained model')
@click.option('--cpu', is_flag=True, 
              help='Training done on the CPU [defaults to GPU]')
@click.option('--shuffle-training', is_flag=True, 
              help='Whether to shuffle the training sample')
@click.option('--generate-weights', is_flag=True, 
              help='Whether to generate just weights and cache them')

def main(view, classes, path, vertex_pass, sample, 
         model_name, epochs, batch_size, validation_pct, seed, 
         cache, cache_path, cpu, shuffle_training, generate_weights):

    if not cpu:
        torch.set_default_device('cuda:0')
        device = torch.device('cuda:0')
        click.echo(f'Using GPU (with name {device}) for training')
    else:
        device = torch.device('cpu')
        click.echo('Using CPU for training')


    # 1. Create the balance map

    BALANCE_MAP = {}
    
    for i, (this_path, pct) in enumerate(sample):
        BALANCE_MAP[f'sample{i}'] = {'dir': this_path, 'fraction': pct}

    click.echo(f'Info: Created sample-balancing map\n{json.dumps(BALANCE_MAP, indent=2)}')
    
    # 2. setting the seed
    set_seed(seed)

    # 3. Paths
    for subdir in ['models', 'stats', 'images', 'cache']:
        dir = f'{cache_path}/{subdir}/pass{vertex_pass}/{view}'
        if not os.path.exists(dir):
            os.makedirs(dir)

    this_cache_path = f'{cache_path}/cache/pass{vertex_pass}/{view}'
    this_model_path = f'{cache_path}/models/pass{vertex_pass}/{view}'

    # 4. DataLoader
    bunch = SegmentationBunch(path, BALANCE_MAP, batch_size=batch_size, valid_pct=validation_pct, device=device)

    # 4a. Classes (and a bunch of caching) 
    
    weights = None
    train_losses, val_losses, batch_losses = None, None, None
    train_accs, val_accs, batch_accs = None, None, None

    initial_epoch = 0

    if cache:
        all_losses_files = glob.glob(f'{this_cache_path}/losses_{model_name}_*.npz')
        all_model_files  = glob.glob(f'{this_model_path}/{model_name}_*.pkl')
        
        if len(all_losses_files) == 0:
            cache=False
            click.echo('Warning: Found no losses files, skipping caching...', err=True)
        elif len(all_model_files) == 0:
            cache=False
            click.echo('Warning: Found no model files, skipping caching...', err=True)
        else:
            latest_epoch_from_losses = int(Path(max(
                all_losses_files, 
                key=lambda p: int(Path(p).stem.split('_')[-1])
            )).stem.split('_')[-1])
            click.echo(f'Info: found losses files, {latest_epoch_from_losses = }')
            
            latest_epoch_from_model = int(Path(max(
                all_model_files, 
                key=lambda p: int(Path(p).stem.split('_')[-1])
            )).stem.split('_')[-1])
            click.echo(f'Info: found model files, {latest_epoch_from_model = }')
            
            latest_epoch = min(latest_epoch_from_model, latest_epoch_from_losses)
            initial_epoch = latest_epoch+1

            if latest_epoch_from_losses != latest_epoch_from_model:
                click.echo(''.join([
                    'Warning: Found mismatch in cached epochs (',
                    f'losses: {latest_epoch_from_losses}, model: {latest_epoch_from_model}). ',
                    f'Resuming from smallest epoch: {latest_epoch}'
                ]))
            else:
                click.echo(f'Info: Resuming from latest epoch n. {latest_epoch}')
    
    if cache:
        weights = np.load(f'{this_cache_path}/weights_{model_name}.npz', 
                          allow_pickle=True)['arr_0']
        weights = np.asarray(weights, dtype=np.float32)
        
        click.echo(f'Info: loaded weights (to {device}) from {this_cache_path}/weights_{model_name}.npz')

        cached_losses = np.load(f'{this_cache_path}/losses_{model_name}_{latest_epoch}.npz', allow_pickle=True)
        train_losses = to_tensor(cached_losses['arr_0'], device)
        val_losses   = to_tensor(cached_losses['arr_1'], device)
        batch_losses = to_tensor(cached_losses['arr_2'], device)
        train_accs   = to_tensor(cached_losses['arr_3'], device)
        val_accs     = to_tensor(cached_losses['arr_4'], device)
        batch_accs   = to_tensor(cached_losses['arr_5'], device)
        
        click.echo(f'Info: loaded losses (to {device}) from {this_cache_path}/weights_{model_name}.npz')
    else:
        train_stats = bunch.count_classes(classes)
        weights = get_class_weights(train_stats)
        np.savez(f'{this_cache_path}/weights_{model_name}.npz', np.asarray(weights, dtype=np.float32))
        click.echo(f'Info: saved weights cache in {this_cache_path}/weights_{model_name}.npz')

        if generate_weights:
            click.echo('Info: Generated weights, returning')
            return
        
        train_losses = torch.zeros(epochs * len(bunch.train_dl), device=device)
        val_losses = torch.zeros(epochs, device=device)
        batch_losses = torch.zeros(len(bunch.valid_dl), device=device)

        train_accs = torch.zeros(epochs * len(bunch.train_dl), device=device)
        val_accs = torch.zeros(epochs, device=device)
        batch_accs = torch.zeros(len(bunch.valid_dl), device=device)
    
    # 5. Model loading...

    model, loss_fn, optim = None, None, None
    
    if cache:
        model, loss_fn, optim = load_model(f'{this_model_path}/{model_name}_{latest_epoch}.pkl', 
                                           classes, weights, device)
        click.echo(f'Info: Loaded model from {this_model_path}/{model_name}_{latest_epoch}.pkl')
    else:
        # Standard model creation
        model, loss_fn, optim = create_model(classes, weights, device)
        click.echo(f'Info: Created model from scratch')

    # 6. Finally training from here down...
    click.echo(f'Info: Will start the training starting from epoch {initial_epoch} to {epochs}')
    
    i = initial_epoch * len(bunch.train_dl)
    
    for e in range(initial_epoch, epochs):

        # Train first 
        model = model.train()
        n_batches = len(bunch.train_dl)

        pbar = tqdm(bunch.train_dl, desc=f'Epoch {e:02d}/{epochs-1:02d}', leave=True)
        
        for b, (x, y) in enumerate(pbar):
            pred = model.forward(x)
            loss = loss_fn(pred, y)
    
            train_losses[i] = loss.item()
            train_accs[i] = accuracy(pred, y, nearby=False)

            pbar.set_postfix(loss=f'{train_losses[i]:.4f}', acc=f'{train_accs[i]:.4f}')
    
            loss.backward()
            optim.step()
            #scheduler.step()
            optim.zero_grad()
            i += 1
            if b == (n_batches - 1):
                save_model(model, x, f'{this_model_path}/{model_name}_{e}')
                tqdm.write(f'Info: Wrote model in cache ({this_model_path}/{model_name}_{e}.pkl)')
    
        # Then validate
        model = model.eval()

        pbar_validation = tqdm(bunch.valid_dl, desc=f'Epoch {e:02d}/{epochs-1:02d} (validation)', leave=True)
        with torch.no_grad():
            for b, (x, y) in enumerate(pbar_validation):
                pred = model.forward(x)
                loss = loss_fn(pred, y)
                
                batch_losses[b] = loss.item()
                batch_accs[b] = accuracy(pred, y, nearby=False)
            val_losses[e] = torch.mean(batch_losses)
            val_accs[e] = torch.mean(batch_accs)

            tqdm.write(f'Info: Summary Epoch {e:02d} | Val Loss: {val_losses[e]:.4f} | Val Acc: {val_accs[e]:.4f}')
        
        np.savez(f'{this_cache_path}/losses_{model_name}_{e}.npz', 
                 train_losses.cpu().numpy(), val_losses.cpu().numpy(), batch_losses.cpu().numpy(), 
                 train_accs.cpu().numpy(), val_accs.cpu().numpy(), batch_accs.cpu().numpy())

    click.echo('Finished training, see you soon for the next adventure!')
    return
    
        
if __name__ == '__main__':
    main()