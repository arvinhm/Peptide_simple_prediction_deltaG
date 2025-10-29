"""
Training Script for Peptide Binding Prediction Model
Includes advanced training techniques and monitoring
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import argparse
import logging
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_processing import prepare_training_data, PeptideDataProcessor
from peptide_binding_model import create_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PeptideBindingDataset(Dataset):
    """Dataset for peptide binding data"""
    
    def __init__(self, samples, protein_seq, max_peptide_len=10):
        self.samples = samples
        self.protein_seq = protein_seq
        self.max_peptide_len = max_peptide_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'peptide_seq': sample['peptide_seq'],
            'peptide_tokens': sample['peptide_tokens'],
            'peptide_features': torch.tensor(sample['peptide_features'], dtype=torch.float32),
            'protein_coords': torch.tensor(sample['protein_coords'], dtype=torch.float32),
            'protein_features': torch.tensor(sample['protein_features'], dtype=torch.float32),
            'interaction_features': torch.tensor(sample['interaction_features'], dtype=torch.float32),
            'delta_g': torch.tensor(sample['delta_g'], dtype=torch.float32)
        }


def collate_batch(batch):
    """Custom collate function"""
    peptide_sequences = [item['peptide_seq'] for item in batch]
    
    # Stack tensors
    peptide_tokens = torch.stack([item['peptide_tokens'] for item in batch])
    peptide_features = torch.stack([item['peptide_features'] for item in batch])
    protein_coords = torch.stack([item['protein_coords'] for item in batch])
    protein_features = torch.stack([item['protein_features'] for item in batch])
    interaction_features = torch.stack([item['interaction_features'] for item in batch])
    targets = torch.stack([item['delta_g'] for item in batch])
    
    return {
        'peptide_sequences': peptide_sequences,
        'peptide_tokens': peptide_tokens,
        'peptide_features': peptide_features,
        'protein_coords': protein_coords,
        'protein_features': protein_features,
        'interaction_features': interaction_features,
        'targets': targets
    }


class ModelTrainer:
    """Advanced trainer with multiple loss functions and techniques"""
    
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate schedulers
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True,
            min_lr=1e-6
        )
        
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.get('warmup_epochs', 5)
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss(delta=1.0)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {'r2': [], 'rmse': [], 'mae': []}
        self.val_metrics = {'r2': [], 'rmse': [], 'mae': []}
    
    def compute_loss(self, predictions, targets, epoch):
        """Compute combined loss with dynamic weighting"""
        
        # Base losses
        mse = self.mse_loss(predictions, targets)
        mae = self.mae_loss(predictions, targets)
        huber = self.huber_loss(predictions, targets)
        
        # Pearson correlation loss (maximize correlation)
        vx = predictions - predictions.mean()
        vy = targets - targets.mean()
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
        corr_loss = 1 - corr
        
        # Dynamic weighting based on epoch
        if epoch < 10:
            # Early training: focus on MSE
            loss = mse + 0.1 * mae + 0.05 * corr_loss
        elif epoch < 30:
            # Mid training: balance all losses
            loss = 0.5 * mse + 0.3 * mae + 0.1 * huber + 0.1 * corr_loss
        else:
            # Late training: focus on correlation
            loss = 0.3 * mse + 0.2 * mae + 0.2 * huber + 0.3 * corr_loss
        
        return loss, {'mse': mse.item(), 'mae': mae.item(), 'huber': huber.item(), 'corr': corr_loss.item()}
    
    def train_epoch(self, train_loader, epoch, protein_seq):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        predictions = []
        targets = []
        loss_components = {'mse': 0, 'mae': 0, 'huber': 0, 'corr': 0}
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch in progress_bar:
            # Move to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(
                batch['peptide_sequences'],
                batch['peptide_tokens'],
                protein_seq,
                batch['protein_coords'],
                batch['peptide_features'],
                batch['interaction_features'],
                batch['protein_features']
            )
            
            # Compute loss
            loss, components = self.compute_loss(outputs, batch['targets'], epoch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            predictions.extend(outputs.detach().cpu().numpy())
            targets.extend(batch['targets'].cpu().numpy())
            
            for key in components:
                loss_components[key] += components[key]
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = epoch_loss / len(train_loader)
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        
        # Average loss components
        for key in loss_components:
            loss_components[key] /= len(train_loader)
        
        return avg_loss, r2, rmse, mae, loss_components
    
    def validate(self, val_loader, epoch, protein_seq):
        """Validate the model"""
        self.model.eval()
        epoch_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                outputs = self.model(
                    batch['peptide_sequences'],
                    batch['peptide_tokens'],
                    protein_seq,
                    batch['protein_coords'],
                    batch['peptide_features'],
                    batch['interaction_features'],
                    batch['protein_features']
                )
                
                loss, _ = self.compute_loss(outputs, batch['targets'], epoch)
                
                # Track metrics
                epoch_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch['targets'].cpu().numpy())
        
        # Calculate metrics
        avg_loss = epoch_loss / len(val_loader)
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        
        return avg_loss, r2, rmse, mae, predictions, targets
    
    def train(self, train_loader, val_loader, protein_seq, n_epochs, save_dir):
        """Complete training loop"""
        best_val_r2 = -float('inf')
        patience_counter = 0
        patience = self.config.get('patience', 15)
        
        for epoch in range(1, n_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{n_epochs}")
            logger.info(f"{'='*50}")
            
            # Training
            train_loss, train_r2, train_rmse, train_mae, loss_components = self.train_epoch(
                train_loader, epoch, protein_seq
            )
            
            # Validation
            val_loss, val_r2, val_rmse, val_mae, val_preds, val_targets = self.validate(
                val_loader, epoch, protein_seq
            )
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics['r2'].append(train_r2)
            self.train_metrics['rmse'].append(train_rmse)
            self.train_metrics['mae'].append(train_mae)
            self.val_metrics['r2'].append(val_r2)
            self.val_metrics['rmse'].append(val_rmse)
            self.val_metrics['mae'].append(val_mae)
            
            # Log metrics
            logger.info(f"Train - Loss: {train_loss:.4f}, R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
            logger.info(f"  Loss components - MSE: {loss_components['mse']:.4f}, MAE: {loss_components['mae']:.4f}, "
                       f"Huber: {loss_components['huber']:.4f}, Corr: {loss_components['corr']:.4f}")
            logger.info(f"Val   - Loss: {val_loss:.4f}, R²: {val_r2:.4f}, RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}")
            
            # Learning rate scheduling
            if epoch <= self.config.get('warmup_epochs', 5):
                self.warmup_scheduler.step()
            else:
                self.scheduler.step(val_loss)
            
            # Save best model
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_r2': val_r2,
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'config': self.config
                }, os.path.join(save_dir, 'best_model.pt'))
                
                logger.info(f"✓ New best model saved! R² = {val_r2:.4f}")
                
                # Save predictions for best model
                self._save_predictions(val_preds, val_targets, save_dir, 'best')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        return best_val_r2
    
    def _save_predictions(self, predictions, targets, save_dir, prefix):
        """Save predictions to file"""
        df = pd.DataFrame({
            'predicted': predictions,
            'actual': targets,
            'error': np.array(predictions) - np.array(targets)
        })
        df.to_csv(os.path.join(save_dir, f'{prefix}_predictions.csv'), index=False)
    
    def plot_training_curves(self, save_dir):
        """Plot and save training curves"""
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # R² curves
        axes[0, 1].plot(epochs, self.train_metrics['r2'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.val_metrics['r2'], 'r-', label='Val', linewidth=2)
        axes[0, 1].axhline(y=0.95, color='g', linestyle='--', label='Target (0.95)', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('R² Score Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # RMSE curves
        axes[1, 0].plot(epochs, self.train_metrics['rmse'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.val_metrics['rmse'], 'r-', label='Val', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE (kcal/mol)')
        axes[1, 0].set_title('Root Mean Square Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # MAE curves
        axes[1, 1].plot(epochs, self.train_metrics['mae'], 'b-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs, self.val_metrics['mae'], 'r-', label='Val', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE (kcal/mol)')
        axes[1, 1].set_title('Mean Absolute Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()


def scatter_plot(predictions, targets, save_path, title="Predicted vs Actual"):
    """Create scatter plot of predictions vs targets"""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(targets, predictions, alpha=0.5, s=20, c='blue', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate metrics
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    
    # Add text box with metrics
    textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f} kcal/mol\nMAE = {mae:.2f} kcal/mol'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.xlabel('Actual ΔG (kcal/mol)', fontsize=12)
    plt.ylabel('Predicted ΔG (kcal/mol)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Peptide Binding Prediction Model')
    parser.add_argument('--pdb_file', type=str, default='PSMA.pdb', help='Protein PDB file')
    parser.add_argument('--csv_file', type=str, default='PSMA_MMGBSA.csv', help='CSV file with peptide data')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.15, help='Test split ratio')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use_se3', action='store_true', default=True, help='Use SE(3) equivariant layers')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging to file
    file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logger.info(f"Starting training with configuration:")
    logger.info(json.dumps(vars(args), indent=2))
    
    # Prepare data
    logger.info("\nPreparing training data...")
    samples, protein_processor, peptide_processor, spatial_extractor = prepare_training_data(
        args.pdb_file, args.csv_file
    )
    
    protein_seq = protein_processor.sequence
    
    # Split data
    indices = np.arange(len(samples))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=args.test_split, random_state=args.seed
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=args.val_split/(1-args.test_split), random_state=args.seed
    )
    
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    
    logger.info(f"Data split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    # Create datasets and loaders
    train_dataset = PeptideBindingDataset(train_samples, protein_seq)
    val_dataset = PeptideBindingDataset(val_samples, protein_seq)
    test_dataset = PeptideBindingDataset(test_samples, protein_seq)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_batch, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_batch, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_batch, pin_memory=True
    )
    
    # Create model
    model_config = {
        'vocab_size': 22,
        'embed_dim': 128,
        'hidden_dim': args.hidden_dim,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'use_se3': args.use_se3,
        'max_peptide_len': 10,
        'max_protein_len': len(protein_seq)
    }
    
    model = create_model(model_config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer_config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_epochs': 5,
        'patience': 15
    }
    
    trainer = ModelTrainer(model, args.device, trainer_config)
    
    # Train model
    logger.info("\nStarting training...")
    best_val_r2 = trainer.train(
        train_loader, val_loader, protein_seq, args.epochs, output_dir
    )
    
    # Plot training curves
    trainer.plot_training_curves(output_dir)
    
    # Load best model for testing
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    logger.info("\n" + "="*50)
    logger.info("Evaluating on test set...")
    logger.info("="*50)
    
    test_loss, test_r2, test_rmse, test_mae, test_preds, test_targets = trainer.validate(
        test_loader, args.epochs, protein_seq
    )
    
    logger.info(f"\nTest Results:")
    logger.info(f"  R² Score: {test_r2:.4f}")
    logger.info(f"  RMSE: {test_rmse:.2f} kcal/mol")
    logger.info(f"  MAE: {test_mae:.2f} kcal/mol")
    
    # Create scatter plots
    scatter_plot(test_preds, test_targets, 
                os.path.join(output_dir, 'test_predictions_scatter.png'),
                "Test Set: Predicted vs Actual ΔG")
    
    # Save test predictions
    test_df = pd.DataFrame({
        'predicted': test_preds,
        'actual': test_targets,
        'error': np.array(test_preds) - np.array(test_targets),
        'abs_error': np.abs(np.array(test_preds) - np.array(test_targets))
    })
    test_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
    
    # Save final results
    results = {
        'best_epoch': int(checkpoint['epoch']),
        'best_val_r2': float(checkpoint['val_r2']),
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'model_params': int(sum(p.numel() for p in model.parameters())),
        'config': vars(args)
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Success message
    if test_r2 >= 0.95:
        logger.info(f"\n{'='*50}")
        logger.info(f"✓✓✓ SUCCESS! Target R² ≥ 0.95 achieved! Test R² = {test_r2:.4f}")
        logger.info(f"{'='*50}")
    else:
        logger.info(f"\nTest R² = {test_r2:.4f}, Target: 0.95")
        logger.info("Consider adjusting hyperparameters or model architecture")
    
    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
