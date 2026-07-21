# network.py

from model import *

import numpy as np
import torch
import torch.optim as opt


def set_seed(seed):
    """Set the various seeds and flags to ensure deterministic performance
    
        Args:
            seed: The random seed
    """
    torch.backends.cudnn.deterministic = True   # Note, can impede performance
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_class_weights(stats):
    """Get the weights for each class
    
        Each class has a weight inversely proportional to the number of instances in the training set
    
        Args:
            stats: The number of instances of each class
        
        Returns:
            The weights for each class
    """
    if np.any(stats == 0.):
        print("Found a class that doesn't appear")
        idx = np.where(stats == 0.)
        stats[idx] = 1
        weights = 1. / stats
        weights[idx] = 0
    else:
        weights = 1. / stats
    return [weight / sum(weights) for weight in weights]


def load_model_only(filename, num_classes, device):
    """Load a model

        Args:
            filename: The name of the file with the pretrained model parameters
            num_classes: The number of classes available to predict
            weights: The weights to apply to the classes
            device: The device on which to run

        Returns:
            A tuple composed (in order) of the model, loss function, and optimiser
    """
    model = UNet(1, n_classes = num_classes, depth = 4, n_filters = 16, y_range = (0, num_classes - 1))
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    return model


def load_model(filename, num_classes, weights, device):
    """Load a model

        Args:
            filename: The name of the file with the pretrained model parameters
            num_classes: The number of classes available to predict
            weights: The weights to apply to the classes
            device: The device on which to run

        Returns:
            A tuple composed (in order) of the model, loss function, and optimiser
    """
    model = UNet(1, n_classes = num_classes, depth = 4, n_filters = 16, y_range = (0, num_classes - 1))
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    loss_fn = nn.CrossEntropyLoss(torch.as_tensor(weights, device=device, dtype=torch.float))
    optim = opt.Adam(model.parameters())
    return model, loss_fn, optim


def save_model(model, input, filename):
    """Save the model
    
        The model is saved as both a pkl file and a TorchScript pt file, which can be loaded via
            model.load_state_dict(torch.load(PATH))
            model.eval()
        
        Args:
            model: The model to save
            input: An example input to the model
            filename: The output filename, without file extension
    """
    torch.save(model.state_dict(), f"{filename}.pkl")

    
def accuracy(pred, truth, nearby=False):
    """Get the network accuracy
    
        Args:
            pred: The network prediction
            truth: The true class
            nearby: Whether to consider adjacent classes acceptable
        
        Returns:
            The accuracy
    """
    target = truth.squeeze(1)
    pred_cls = pred.argmax(dim=1)
    mask = (target != 0)
    if nearby:
        result = abs(pred_cls[mask] - target[mask]) <= 1
    else:
        result = pred_cls[mask] == target[mask]
    return result.float().mean()


def create_model(num_classes, weights, device):
    """Create the model

        Args:
            num_classes: The number of classes available to predict
            weights: The weights to apply to the classes
            device: The device on which to run

        Returns:
            A tuple composed (in order) of the model, loss function, and optimiser
    """
    model = UNet(1, n_classes = num_classes, depth = 4, n_filters = 16, y_range = (0, num_classes - 1))
    loss_fn = nn.CrossEntropyLoss(torch.as_tensor(weights, device=device, dtype=torch.float))
    optim = opt.Adam(model.parameters())
    return model, loss_fn, optim
