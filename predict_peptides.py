"""
Peptide Prediction and Generation Script
Generate new peptide candidates and predict their binding affinities
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import argparse
import json
import os
from itertools import combinations, product
import logging
from tqdm import tqdm

from data_processing import (
    ProteinStructureProcessor, 
    PeptideDataProcessor, 
    SpatialFeatureExtractor,
    AA_TO_INDEX,
    AA_PROPERTIES
)
from peptide_binding_model import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard amino acids for generation
STANDARD_AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Amino acids by properties for guided generation
AA_BY_PROPERTY = {
    'hydrophobic': ['A', 'I', 'L', 'M', 'F', 'W', 'V'],
    'hydrophilic': ['S', 'T', 'N', 'Q', 'C'],
    'charged_positive': ['R', 'K', 'H'],
    'charged_negative': ['D', 'E'],
    'aromatic': ['F', 'W', 'Y', 'H'],
    'small': ['G', 'A', 'S'],
    'large': ['W', 'Y', 'F', 'R', 'K']
}


class PeptideGenerator:
    """Generate peptide candidates using various strategies"""
    
    def __init__(self, existing_peptides: List[str]):
        self.existing_peptides = set(existing_peptides)
        self.top_peptides = self._get_top_peptides(existing_peptides)
    
    def _get_top_peptides(self, peptides: List[str], top_n: int = 20) -> List[str]:
        """Get top performing peptides from existing data"""
        # This would be based on actual delta G values in practice
        return peptides[:top_n] if len(peptides) > top_n else peptides
    
    def generate_mutations(self, base_peptide: str, n_mutations: int = 1) -> List[str]:
        """Generate single or multiple point mutations"""
        candidates = []
        positions = list(range(len(base_peptide)))
        
        for mut_positions in combinations(positions, n_mutations):
            for aas in product(STANDARD_AA, repeat=n_mutations):
                new_peptide = list(base_peptide)
                for pos, aa in zip(mut_positions, aas):
                    new_peptide[pos] = aa
                new_peptide = ''.join(new_peptide)
                
                if new_peptide not in self.existing_peptides:
                    candidates.append(new_peptide)
        
        return candidates
    
    def generate_extensions(self, base_peptide: str, max_length: int = 8) -> List[str]:
        """Extend peptides at N or C terminus"""
        candidates = []
        
        if len(base_peptide) >= max_length:
            return candidates
        
        # N-terminal extensions
        for aa in STANDARD_AA:
            new_peptide = aa + base_peptide
            if new_peptide not in self.existing_peptides:
                candidates.append(new_peptide)
        
        # C-terminal extensions
        for aa in STANDARD_AA:
            new_peptide = base_peptide + aa
            if new_peptide not in self.existing_peptides:
                candidates.append(new_peptide)
        
        return candidates
    
    def generate_property_based(self, length: int, properties: List[str]) -> List[str]:
        """Generate peptides with specific property patterns"""
        candidates = []
        
        # Define patterns based on properties
        patterns = []
        if 'amphipathic' in properties:
            # Alternating hydrophobic and hydrophilic
            patterns.append(['hydrophobic', 'hydrophilic'] * (length // 2))
        
        if 'charged' in properties:
            # Mix of charged and neutral
            patterns.append(['charged_positive', 'hydrophobic', 'charged_negative'] * (length // 3))
        
        if 'aromatic' in properties:
            # Include aromatic residues
            patterns.append(['aromatic', 'small', 'aromatic'] * (length // 3))
        
        # Generate peptides based on patterns
        for pattern in patterns:
            pattern = pattern[:length]  # Trim to desired length
            
            # Generate a few variants for each pattern
            for _ in range(10):
                peptide = ''
                for prop in pattern:
                    if prop in AA_BY_PROPERTY:
                        peptide += np.random.choice(AA_BY_PROPERTY[prop])
                    else:
                        peptide += np.random.choice(STANDARD_AA)
                
                if peptide not in self.existing_peptides and len(peptide) == length:
                    candidates.append(peptide)
        
        return candidates
    
    def generate_scaffolds(self, base_peptides: List[str]) -> List[str]:
        """Generate scaffolds by combining successful motifs"""
        candidates = []
        
        # Extract common motifs from successful peptides
        motifs = self._extract_motifs(base_peptides)
        
        # Combine motifs to create new peptides
        for motif1, motif2 in combinations(motifs, 2):
            # Direct concatenation
            if len(motif1) + len(motif2) <= 8:
                new_peptide = motif1 + motif2
                if new_peptide not in self.existing_peptides:
                    candidates.append(new_peptide)
            
            # With linker
            for linker in ['G', 'GG', 'S', 'GS']:
                if len(motif1) + len(motif2) + len(linker) <= 8:
                    new_peptide = motif1 + linker + motif2
                    if new_peptide not in self.existing_peptides:
                        candidates.append(new_peptide)
        
        return candidates
    
    def _extract_motifs(self, peptides: List[str], min_length: int = 2, max_length: int = 4) -> List[str]:
        """Extract common motifs from peptides"""
        motif_counts = {}
        
        for peptide in peptides:
            for length in range(min_length, min(max_length + 1, len(peptide) + 1)):
                for i in range(len(peptide) - length + 1):
                    motif = peptide[i:i + length]
                    motif_counts[motif] = motif_counts.get(motif, 0) + 1
        
        # Return motifs that appear in multiple peptides
        common_motifs = [motif for motif, count in motif_counts.items() if count >= 2]
        return common_motifs[:20]  # Return top 20 motifs
    
    def generate_diverse_library(self, n_peptides: int = 1000, 
                                min_length: int = 3, max_length: int = 8) -> List[str]:
        """Generate a diverse library of peptide candidates"""
        all_candidates = []
        
        # 1. Mutations of top performers
        logger.info("Generating mutations of top performers...")
        for peptide in self.top_peptides[:10]:
            all_candidates.extend(self.generate_mutations(peptide, 1))
            all_candidates.extend(self.generate_mutations(peptide, 2))
        
        # 2. Extensions of successful short peptides
        logger.info("Generating extensions...")
        short_peptides = [p for p in self.top_peptides if len(p) <= 5]
        for peptide in short_peptides[:10]:
            all_candidates.extend(self.generate_extensions(peptide, max_length))
        
        # 3. Property-based generation
        logger.info("Generating property-based peptides...")
        for length in range(min_length, max_length + 1):
            all_candidates.extend(self.generate_property_based(length, ['amphipathic']))
            all_candidates.extend(self.generate_property_based(length, ['charged']))
            all_candidates.extend(self.generate_property_based(length, ['aromatic']))
        
        # 4. Scaffold-based generation
        logger.info("Generating scaffold-based peptides...")
        all_candidates.extend(self.generate_scaffolds(self.top_peptides[:15]))
        
        # 5. Random generation to fill library
        logger.info("Generating random peptides...")
        while len(all_candidates) < n_peptides:
            length = np.random.randint(min_length, max_length + 1)
            peptide = ''.join(np.random.choice(STANDARD_AA) for _ in range(length))
            if peptide not in self.existing_peptides and peptide not in all_candidates:
                all_candidates.append(peptide)
        
        # Remove duplicates and limit to requested number
        unique_candidates = list(set(all_candidates))
        np.random.shuffle(unique_candidates)
        
        return unique_candidates[:n_peptides]


class PeptidePredictor:
    """Predict binding affinities for peptide candidates"""
    
    def __init__(self, model_path: str, pdb_file: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_config = checkpoint.get('config', {
            'vocab_size': 22,
            'embed_dim': 128,
            'hidden_dim': 256,
            'n_heads': 8,
            'n_layers': 4,
            'dropout': 0.1,
            'use_se3': True
        })
        
        # Process protein structure
        self.protein_processor = ProteinStructureProcessor(pdb_file)
        self.protein_seq, self.protein_coords, self.protein_features = self.protein_processor.process_protein()
        
        # Create model
        self.model_config['max_protein_len'] = len(self.protein_seq)
        self.model = create_model(self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Spatial feature extractor
        self.spatial_extractor = SpatialFeatureExtractor(self.protein_coords, self.protein_features)
        self.binding_sites = self.spatial_extractor.get_binding_site_candidates()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Protein processed: {len(self.protein_seq)} residues")
    
    def predict_batch(self, peptides: List[str], batch_size: int = 32) -> np.ndarray:
        """Predict binding affinities for a batch of peptides"""
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(peptides), batch_size):
                batch_peptides = peptides[i:i + batch_size]
                batch_predictions = self._predict_single_batch(batch_peptides)
                predictions.extend(batch_predictions)
        
        return np.array(predictions)
    
    def _predict_single_batch(self, peptides: List[str]) -> List[float]:
        """Predict for a single batch"""
        batch_size = len(peptides)
        
        # Prepare batch data
        peptide_tokens = []
        peptide_features = []
        interaction_features = []
        
        for peptide in peptides:
            # Tokenize
            tokens = [AA_TO_INDEX.get(aa, 21) for aa in peptide]
            tokens.extend([0] * (10 - len(tokens)))  # Pad to max length
            peptide_tokens.append(tokens)
            
            # Compute features
            pep_features = self._compute_peptide_features(peptide)
            peptide_features.append(pep_features)
            
            # Compute interaction features with best binding site
            int_features = self.spatial_extractor.compute_interaction_features(
                pep_features, self.binding_sites[0]
            )
            interaction_features.append(int_features)
        
        # Convert to tensors
        peptide_tokens = torch.tensor(peptide_tokens, dtype=torch.long).to(self.device)
        peptide_features = torch.tensor(np.array(peptide_features), dtype=torch.float32).to(self.device)
        interaction_features = torch.tensor(np.array(interaction_features), dtype=torch.float32).to(self.device)
        
        # Repeat protein coords for batch
        protein_coords = torch.tensor(self.protein_coords, dtype=torch.float32).unsqueeze(0)
        protein_coords = protein_coords.repeat(batch_size, 1, 1).to(self.device)
        
        protein_features = torch.tensor(self.protein_features, dtype=torch.float32).unsqueeze(0)
        protein_features = protein_features.repeat(batch_size, 1, 1).to(self.device)
        
        # Predict
        predictions = self.model(
            peptides,
            peptide_tokens,
            self.protein_seq,
            protein_coords,
            peptide_features,
            interaction_features,
            protein_features
        )
        
        return predictions.cpu().numpy().tolist()
    
    def _compute_peptide_features(self, peptide: str) -> np.ndarray:
        """Compute features for a peptide"""
        features = []
        
        # Length
        features.append(len(peptide))
        
        # Composition
        aa_counts = np.zeros(20)
        for aa in peptide:
            if aa in AA_TO_INDEX:
                idx = AA_TO_INDEX[aa] - 1
                if idx < 20:
                    aa_counts[idx] += 1
        features.extend(aa_counts / max(len(peptide), 1))
        
        # Average properties
        props = []
        for aa in peptide:
            if aa in AA_PROPERTIES:
                props.append(AA_PROPERTIES[aa])
        
        if props:
            avg_props = np.mean(props, axis=0)
            features.extend(avg_props)
        else:
            features.extend([0] * 8)
        
        # Terminal properties
        if len(peptide) > 0:
            n_term = AA_PROPERTIES.get(peptide[0], [0] * 8)
            c_term = AA_PROPERTIES.get(peptide[-1], [0] * 8)
            features.extend(n_term)
            features.extend(c_term)
        else:
            features.extend([0] * 16)
        
        return np.array(features, dtype=np.float32)
    
    def predict_with_uncertainty(self, peptides: List[str], n_forward: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimation using dropout"""
        self.model.train()  # Enable dropout
        
        all_predictions = []
        
        for _ in range(n_forward):
            predictions = self.predict_batch(peptides)
            all_predictions.append(predictions)
        
        self.model.eval()
        
        all_predictions = np.array(all_predictions)
        mean_predictions = all_predictions.mean(axis=0)
        std_predictions = all_predictions.std(axis=0)
        
        return mean_predictions, std_predictions


def main():
    parser = argparse.ArgumentParser(description='Generate and predict peptide candidates')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--pdb_file', type=str, default='PSMA.pdb', help='Protein PDB file')
    parser.add_argument('--csv_file', type=str, default='PSMA_MMGBSA.csv', help='Existing peptide data')
    parser.add_argument('--n_candidates', type=int, default=1000, help='Number of candidates to generate')
    parser.add_argument('--min_length', type=int, default=3, help='Minimum peptide length')
    parser.add_argument('--max_length', type=int, default=8, help='Maximum peptide length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction')
    parser.add_argument('--output_file', type=str, default='predictions.csv', help='Output CSV file')
    parser.add_argument('--with_uncertainty', action='store_true', help='Include uncertainty estimates')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load existing peptide data
    logger.info("Loading existing peptide data...")
    existing_data = pd.read_csv(args.csv_file)
    existing_peptides = existing_data['title'].tolist()
    
    # Generate candidates
    logger.info(f"Generating {args.n_candidates} peptide candidates...")
    generator = PeptideGenerator(existing_peptides)
    candidates = generator.generate_diverse_library(
        args.n_candidates, 
        args.min_length, 
        args.max_length
    )
    
    logger.info(f"Generated {len(candidates)} unique candidates")
    
    # Initialize predictor
    logger.info("Initializing predictor...")
    predictor = PeptidePredictor(args.model_path, args.pdb_file, args.device)
    
    # Predict binding affinities
    logger.info("Predicting binding affinities...")
    
    if args.with_uncertainty:
        predictions, uncertainties = predictor.predict_with_uncertainty(candidates)
    else:
        predictions = predictor.predict_batch(candidates, args.batch_size)
        uncertainties = np.zeros_like(predictions)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'peptide': candidates,
        'predicted_dG': predictions,
        'uncertainty': uncertainties,
        'length': [len(p) for p in candidates]
    })
    
    # Sort by predicted affinity (more negative = better)
    results_df = results_df.sort_values('predicted_dG', ascending=True)
    
    # Add ranking
    results_df['rank'] = range(1, len(results_df) + 1)
    
    # Calculate peptide properties
    logger.info("Calculating peptide properties...")
    properties = []
    for peptide in tqdm(results_df['peptide'], desc="Computing properties"):
        prop_dict = {
            'n_charged': sum(1 for aa in peptide if aa in 'RKDE'),
            'n_aromatic': sum(1 for aa in peptide if aa in 'FWY'),
            'n_hydrophobic': sum(1 for aa in peptide if aa in 'AILMFWV'),
            'charge': sum(1 for aa in peptide if aa in 'RK') - sum(1 for aa in peptide if aa in 'DE')
        }
        properties.append(prop_dict)
    
    prop_df = pd.DataFrame(properties)
    results_df = pd.concat([results_df, prop_df], axis=1)
    
    # Save results
    results_df.to_csv(args.output_file, index=False)
    logger.info(f"Results saved to {args.output_file}")
    
    # Print top candidates
    logger.info("\nTop 20 Predicted Peptides:")
    logger.info("-" * 60)
    for idx, row in results_df.head(20).iterrows():
        logger.info(f"Rank {row['rank']:3d}: {row['peptide']:10s} | "
                   f"ΔG: {row['predicted_dG']:8.2f} ± {row['uncertainty']:5.2f} kcal/mol")
    
    # Statistics
    logger.info("\nStatistics:")
    logger.info(f"Best predicted ΔG: {results_df['predicted_dG'].min():.2f} kcal/mol")
    logger.info(f"Average predicted ΔG: {results_df['predicted_dG'].mean():.2f} kcal/mol")
    logger.info(f"Number of candidates < -100 kcal/mol: {(results_df['predicted_dG'] < -100).sum()}")
    logger.info(f"Number of candidates < -120 kcal/mol: {(results_df['predicted_dG'] < -120).sum()}")
    
    # Length distribution of top candidates
    top_100 = results_df.head(100)
    length_dist = top_100['length'].value_counts().sort_index()
    logger.info("\nLength distribution of top 100 candidates:")
    for length, count in length_dist.items():
        logger.info(f"  Length {length}: {count} peptides")
    
    # Save top candidates separately
    top_candidates_file = args.output_file.replace('.csv', '_top100.csv')
    results_df.head(100).to_csv(top_candidates_file, index=False)
    logger.info(f"\nTop 100 candidates saved to {top_candidates_file}")


if __name__ == "__main__":
    main()
