# Peptide Binding Affinity Prediction Model

## 🧬 Advanced AI Model for Peptide-Protein Binding Prediction

This project implements a state-of-the-art deep learning model to predict peptide binding affinity (ΔG) to proteins using spatial features, attention mechanisms, and physicochemical properties. The model achieves high accuracy (target R² > 0.95) by incorporating protein structure information and learning from duplicate peptide measurements at different binding sites.

## 🎯 Key Features

- **SE(3)-Equivariant Architecture**: Leverages 3D protein structure for spatial understanding
- **Property-Aware Tokenization**: Incorporates physicochemical properties of amino acids
- **Multi-Head Attention**: Cross-attention between peptide and protein sequences
- **Spatial Learning from Duplicates**: Learns binding site preferences from duplicate measurements
- **Peptide Generation**: Intelligent generation of new peptide candidates
- **Uncertainty Estimation**: Provides confidence intervals for predictions

## 📁 Project Structure

```
peptide_pred/
├── PSMA.pdb                    # Protein structure file
├── PSMA_MMGBSA.csv             # Peptide binding data (sequence, ΔG)
├── data_processing.py          # Data processing and feature extraction
├── peptide_binding_model.py   # Model architecture
├── train_model.py             # Training script
├── predict_peptides.py        # Prediction and peptide generation
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository (if applicable)
cd peptide_pred

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py \
    --pdb_file PSMA.pdb \
    --csv_file PSMA_MMGBSA.csv \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --hidden_dim 256 \
    --output_dir results
```

### 3. Generate and Predict New Peptides

```bash
python predict_peptides.py \
    --model_path results/run_*/best_model.pt \
    --pdb_file PSMA.pdb \
    --csv_file PSMA_MMGBSA.csv \
    --n_candidates 1000 \
    --output_file predictions.csv \
    --with_uncertainty
```

## 📊 Model Architecture

### Core Components

1. **Peptide Encoder**
   - Property-aware amino acid embeddings
   - Positional encoding
   - Self-attention layers
   - Incorporates 8 physicochemical properties per amino acid

2. **Protein Encoder**
   - SE(3)-equivariant layers for 3D structure processing
   - Spatial coordinate refinement
   - Feature fusion with sequence information

3. **Cross-Attention Module**
   - Bidirectional attention between peptide and protein
   - Multiple attention heads (default: 8)
   - Layer normalization and dropout

4. **Interaction Prediction Head**
   - Combines global and local features
   - Spatial feature processing
   - Binding site attention mechanism

### Model Specifications

- **Input**: Peptide sequence (3-10 AA), Protein structure (PDB)
- **Output**: Binding affinity ΔG (kcal/mol)
- **Parameters**: ~2-5M (configurable)
- **Architecture**: Transformer-based with SE(3) equivariance

## 🔬 Technical Innovations

### 1. Spatial Learning from Duplicates
The model learns from duplicate peptide measurements at different binding sites:
- Each duplicate represents different spatial orientation/binding location
- Helps the model understand preferred binding sites
- Improves generalization to new peptides

### 2. Property-Aware Tokenization
Each amino acid is represented by:
- Token ID (1-20 for standard AAs)
- 8 normalized physicochemical properties:
  - Hydrophobicity
  - Volume
  - Charge
  - Polarity
  - Aromaticity
  - H-bond donor capacity
  - H-bond acceptor capacity
  - Flexibility

### 3. SE(3)-Equivariant Processing
- Maintains rotational and translational invariance
- Processes 3D coordinates directly
- Refines spatial features through multiple layers

### 4. Dynamic Loss Function
Training uses a combination of:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss (robust to outliers)
- Correlation Loss (maximize Pearson correlation)

Weights are dynamically adjusted during training phases.

## 📈 Training Details

### Data Processing

1. **Protein Structure Processing**
   - Extract Cα coordinates from PDB
   - Normalize coordinates (center at origin, unit variance)
   - Extract local environment features (10Å radius)

2. **Peptide Feature Extraction**
   - Sequence tokenization
   - Property computation
   - Terminal residue analysis
   - Interaction feature calculation

3. **Binding Site Detection**
   - Automatic pocket detection
   - Surface accessibility analysis
   - Feature diversity scoring

### Training Strategy

- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 5e-4 with ReduceLROnPlateau scheduler
- **Warm-up**: 5 epochs with linear warm-up
- **Early Stopping**: Patience of 15 epochs
- **Batch Size**: 32 (configurable)
- **Gradient Clipping**: Max norm of 1.0

### Evaluation Metrics

- **R² Score**: Target ≥ 0.95
- **RMSE**: Root Mean Square Error (kcal/mol)
- **MAE**: Mean Absolute Error (kcal/mol)
- **Pearson Correlation**: Linear correlation coefficient

## 🧪 Peptide Generation Strategies

The `predict_peptides.py` script implements multiple generation strategies:

### 1. Mutation-Based
- Single and double point mutations of top performers
- Preserves successful scaffolds

### 2. Extension-Based
- N-terminal and C-terminal additions
- Gradual elongation of short peptides

### 3. Property-Based
- Amphipathic patterns
- Charge distribution optimization
- Aromatic residue placement

### 4. Scaffold Combination
- Extracts successful motifs
- Combines motifs with appropriate linkers
- Creates novel combinations

### 5. Diversity Enhancement
- Random generation for exploration
- Ensures chemical diversity
- Avoids overfitting to training data

## 📋 Command-Line Arguments

### train_model.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--pdb_file` | PSMA.pdb | Protein structure file |
| `--csv_file` | PSMA_MMGBSA.csv | Peptide binding data |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 5e-4 | Initial learning rate |
| `--hidden_dim` | 256 | Hidden dimension size |
| `--n_heads` | 8 | Number of attention heads |
| `--n_layers` | 4 | Number of transformer layers |
| `--dropout` | 0.1 | Dropout rate |
| `--val_split` | 0.15 | Validation split ratio |
| `--test_split` | 0.15 | Test split ratio |

### predict_peptides.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | Required | Path to trained model |
| `--n_candidates` | 1000 | Number of peptides to generate |
| `--min_length` | 3 | Minimum peptide length |
| `--max_length` | 8 | Maximum peptide length |
| `--with_uncertainty` | False | Include uncertainty estimates |
| `--batch_size` | 32 | Prediction batch size |

## 📊 Expected Results

### Training Performance
- **Training Time**: ~1-3 hours on GPU (depending on dataset size)
- **Convergence**: Typically within 50-70 epochs
- **Target R²**: ≥ 0.95 on test set
- **RMSE**: < 5 kcal/mol (dataset dependent)

### Prediction Output
The prediction script generates a CSV with:
- Peptide sequence
- Predicted ΔG (kcal/mol)
- Uncertainty estimate (if requested)
- Peptide properties (charge, hydrophobicity, etc.)
- Ranking by predicted affinity

## 🔧 Customization

### Modifying Model Architecture

Edit `peptide_binding_model.py`:

```python
model_config = {
    'vocab_size': 22,
    'embed_dim': 128,
    'hidden_dim': 256,  # Increase for more capacity
    'n_heads': 8,       # More heads for complex patterns
    'n_layers': 4,      # Deeper network
    'use_se3': True,    # Toggle SE(3) equivariance
}
```

### Adding Custom Features

Extend the `AA_PROPERTIES` dictionary in `data_processing.py`:

```python
AA_PROPERTIES['A'] = [1.8, 88.6, 0.0, ...]  # Add more properties
```

### Custom Loss Functions

Modify `compute_loss()` in `train_model.py`:

```python
def compute_loss(self, predictions, targets, epoch):
    # Add custom loss components
    custom_loss = your_custom_function(predictions, targets)
    return base_loss + weight * custom_loss
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Decrease hidden dimensions
   - Use gradient accumulation

2. **Poor Convergence**
   - Adjust learning rate
   - Increase warm-up epochs
   - Check data quality

3. **Overfitting**
   - Increase dropout
   - Add weight decay
   - Reduce model size
   - Augment training data

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{peptide_binding_2024,
  title={Advanced Peptide Binding Affinity Prediction Model},
  author={Your Name},
  year={2024},
  url={https://github.com/yourrepo}
}
```

## 📝 License

This project is provided for research and educational purposes.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📧 Contact

For questions or collaborations, please contact [your-email]

## 🙏 Acknowledgments

- BioPython for PDB parsing
- PyTorch for deep learning framework
- Inspired by recent advances in protein-peptide interaction prediction

---

**Note**: This model is designed for research purposes. Always validate predictions experimentally before making critical decisions.
