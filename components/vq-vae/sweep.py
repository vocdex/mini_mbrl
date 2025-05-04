import argparse
import wandb
from box import Box
from vqvae_cifar10 import Config, VQVAE, get_data_loaders, seed_everything
from utils import train_model

def sweep_train_function():
    """Function called by wandb.agent for each sweep run"""
    run = wandb.init()
    config = Config()
    for key, value in run.config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    cfg = Box(config.to_dict())
    seed_everything(cfg.seed)
    train_loader, test_loader = get_data_loaders(cfg)

    model = VQVAE(cfg).to(cfg.device)

    run.watch(model, log="all")

    model, _, _ = train_model(model, train_loader, test_loader, cfg, wandb_run=run)
    
    model.save(config.save_dir / f"vqvae_sweep_{run.id}.pt")


def run_sweep(sweep_config, project_name, count=10):
    """Create and run a wandb sweep with the given configuration"""
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, sweep_train_function, count=count)
    return sweep_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VQ-VAE Hyperparameter Sweep')
    parser.add_argument('--count', type=int, default=24,
                        help='Number of sweep runs to execute')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs per sweep run')
    args = parser.parse_args()
    
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'test/loss',
            'goal': 'minimize'
        },
        'parameters': {
            'num_embeddings': {
                'values': [256, 512]
            },
            'num_epochs': {
                'value': args.epochs
            },
            'commitment_cost':{
                'values': [0.25, 0.50]
            },
            'lr_patience': {
                'values': [5, 10]
            },
        }
    }
    
    run_sweep(sweep_config, "vqvae-cifar10-sweep", count=args.count)