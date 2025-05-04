# vqvae_utils.py - Shared components between main script and sweep
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, train_loader, test_loader, config, wandb_run=None):
    """Generic training function used by both normal training and sweeps"""
    # Create optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        factor=config.lr_factor, 
        patience=config.lr_patience, 
        verbose=True
    )
    
    # Import the training function (but only at function call time to avoid circular imports)
    from vqvae_cifar10 import train
    
    # Train the model
    print(f"Starting training on {config.device}")
    model = train(model, train_loader, test_loader, optimizer, scheduler, config)
    
    return model, optimizer, scheduler