"""
Test script to verify all components are working correctly
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        from Bio.PDB import PDBParser
        print("✓ BioPython imported successfully")
    except ImportError as e:
        print(f"✗ BioPython import failed: {e}")
        print("  Please install: pip install biopython")
        return False
    
    try:
        from sklearn.model_selection import train_test_split
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    return True


def test_custom_modules():
    """Test that custom modules can be imported"""
    print("\nTesting custom modules...")
    
    try:
        from data_processing import (
            ProteinStructureProcessor,
            PeptideDataProcessor,
            SpatialFeatureExtractor,
            prepare_training_data
        )
        print("✓ data_processing.py imported successfully")
    except ImportError as e:
        print(f"✗ data_processing.py import failed: {e}")
        return False
    
    try:
        from peptide_binding_model import (
            PeptideBindingModel,
            create_model,
            PeptideEncoder,
            ProteinEncoder
        )
        print("✓ peptide_binding_model.py imported successfully")
    except ImportError as e:
        print(f"✗ peptide_binding_model.py import failed: {e}")
        return False
    
    return True


def test_data_files():
    """Test that required data files exist"""
    print("\nTesting data files...")
    
    import os
    
    files_to_check = [
        ("PSMA.pdb", "Protein structure file"),
        ("PSMA_MMGBSA.csv", "Peptide binding data")
    ]
    
    all_exist = True
    for filename, description in files_to_check:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✓ {filename} exists ({size:,} bytes) - {description}")
        else:
            print(f"✗ {filename} not found - {description}")
            all_exist = False
    
    return all_exist


def test_basic_functionality():
    """Test basic functionality of the modules"""
    print("\nTesting basic functionality...")
    
    try:
        # Test model creation
        from peptide_binding_model import create_model
        config = {
            'vocab_size': 22,
            'embed_dim': 64,
            'hidden_dim': 128,
            'n_heads': 4,
            'n_layers': 2
        }
        model = create_model(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created successfully ({param_count:,} parameters)")
        
        # Test basic data processing
        import pandas as pd
        df = pd.read_csv("PSMA_MMGBSA.csv")
        print(f"✓ CSV loaded successfully ({len(df)} rows)")
        
        # Check for duplicates (important for spatial learning)
        duplicates = df[df.duplicated(subset=['title'], keep=False)]
        unique_peptides = df['title'].nunique()
        print(f"✓ Found {unique_peptides} unique peptides with {len(duplicates)} duplicate measurements")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_device():
    """Test CUDA availability"""
    print("\nTesting compute device...")
    
    import torch
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("✓ CUDA not available, will use CPU")
        print("  Note: Training will be slower on CPU")
    
    return True


def run_mini_training():
    """Run a mini training session to verify everything works"""
    print("\nRunning mini training test...")
    
    try:
        import torch
        import numpy as np
        from data_processing import prepare_training_data
        from peptide_binding_model import create_model
        
        # Load a small subset of data
        print("  Loading data subset...")
        samples, protein_processor, peptide_processor, spatial_extractor = prepare_training_data(
            "PSMA.pdb", "PSMA_MMGBSA.csv"
        )
        
        # Use only first 10 samples for testing
        test_samples = samples[:10]
        
        # Create a small model
        print("  Creating model...")
        config = {
            'vocab_size': 22,
            'embed_dim': 32,
            'hidden_dim': 64,
            'n_heads': 2,
            'n_layers': 1,
            'max_protein_len': len(protein_processor.sequence)
        }
        model = create_model(config)
        
        # Test forward pass
        print("  Testing forward pass...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Prepare a batch
        peptide_sequences = [s['peptide_seq'] for s in test_samples[:4]]
        peptide_tokens = torch.stack([s['peptide_tokens'] for s in test_samples[:4]]).to(device)
        protein_coords = torch.tensor(test_samples[0]['protein_coords']).unsqueeze(0).repeat(4, 1, 1).to(device)
        peptide_features = torch.tensor(np.array([s['peptide_features'] for s in test_samples[:4]])).to(device)
        interaction_features = torch.tensor(np.array([s['interaction_features'] for s in test_samples[:4]])).to(device)
        protein_features = torch.tensor(test_samples[0]['protein_features']).unsqueeze(0).repeat(4, 1, 1).to(device)
        
        # Forward pass
        with torch.no_grad():
            predictions = model(
                peptide_sequences,
                peptide_tokens,
                protein_processor.sequence,
                protein_coords,
                peptide_features,
                interaction_features,
                protein_features
            )
        
        print(f"✓ Mini training test successful! Predictions shape: {predictions.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Mini training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("PEPTIDE BINDING PREDICTION - SETUP TEST")
    print("="*60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n⚠ Please install missing dependencies: pip install -r requirements.txt")
    
    # Test custom modules
    if not test_custom_modules():
        all_passed = False
        print("\n⚠ Custom modules not found. Make sure all .py files are in the current directory")
    
    # Test data files
    if not test_data_files():
        all_passed = False
        print("\n⚠ Data files missing. Make sure PSMA.pdb and PSMA_MMGBSA.csv are present")
    
    # Test device
    test_device()
    
    # Test basic functionality
    if all_passed:
        if not test_basic_functionality():
            all_passed = False
    
    # Run mini training if everything else passed
    if all_passed:
        if not run_mini_training():
            all_passed = False
    
    # Final summary
    print("\n" + "="*60)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED! Ready to train the model.")
        print("\nNext steps:")
        print("1. Train the model: python train_model.py")
        print("2. Generate predictions: python predict_peptides.py --model_path results/*/best_model.pt")
    else:
        print("✗ Some tests failed. Please fix the issues above before training.")
    print("="*60)


if __name__ == "__main__":
    main()
