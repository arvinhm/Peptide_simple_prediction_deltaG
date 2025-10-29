"""
Data Processing Module for Peptide-Protein Binding Prediction
Handles PDB structure extraction, CSV processing, and tokenization
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from Bio.PDB import PDBParser, is_aa
import torch
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 3-letter to 1-letter amino acid mapping
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Amino acid to index mapping
AA_TO_INDEX = {
    "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9,
    "L": 10, "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17,
    "V": 18, "W": 19, "Y": 20, "X": 21  # X for unknown
}

# Enhanced amino acid properties for spatial understanding
AA_PROPERTIES = {
    # [hydrophobicity, volume, charge, polarity, aromaticity, H-bond donor, H-bond acceptor, flexibility]
    'A': [1.8, 88.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35],
    'R': [-4.5, 173.4, 1.0, 1.0, 0.0, 4.0, 1.0, 0.53],
    'N': [-3.5, 114.1, 0.0, 1.0, 0.0, 2.0, 2.0, 0.46],
    'D': [-3.5, 111.1, -1.0, 1.0, 0.0, 1.0, 3.0, 0.51],
    'C': [2.5, 108.5, 0.0, 0.25, 0.0, 1.0, 1.0, 0.35],
    'Q': [-3.5, 143.8, 0.0, 1.0, 0.0, 2.0, 2.0, 0.49],
    'E': [-3.5, 138.4, -1.0, 1.0, 0.0, 1.0, 3.0, 0.50],
    'G': [-0.4, 60.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.54],
    'H': [-3.2, 153.2, 0.1, 1.0, 1.0, 2.0, 1.0, 0.32],
    'I': [4.5, 166.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.46],
    'L': [3.8, 166.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.37],
    'K': [-3.9, 168.6, 1.0, 1.0, 0.0, 3.0, 1.0, 0.47],
    'M': [1.9, 162.9, 0.0, 0.0, 0.0, 0.0, 1.0, 0.30],
    'F': [2.8, 189.9, 0.0, 0.0, 1.0, 0.0, 0.0, 0.31],
    'P': [-1.6, 112.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.51],
    'S': [-0.8, 89.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.51],
    'T': [-0.7, 116.1, 0.0, 1.0, 0.0, 1.0, 1.0, 0.44],
    'W': [-0.9, 227.8, 0.0, 0.1, 1.0, 1.0, 1.0, 0.31],
    'Y': [-1.3, 193.6, 0.0, 1.0, 1.0, 1.0, 1.0, 0.42],
    'V': [4.2, 140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.39],
    'X': [0.0, 120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40],  # Default for unknown
}


class ProteinStructureProcessor:
    """Process protein PDB file to extract structure and spatial information"""
    
    def __init__(self, pdb_file: str):
        self.pdb_file = pdb_file
        self.parser = PDBParser(QUIET=True)
        self.structure = None
        self.sequence = None
        self.coordinates = None
        self.residue_features = None
        
    def process_protein(self) -> Tuple[str, np.ndarray, np.ndarray]:
        """
        Extract protein sequence and coordinates from PDB
        Returns: (sequence, coordinates, residue_features)
        """
        structure = self.parser.get_structure("protein", self.pdb_file)
        
        sequence_chars = []
        coordinates = []
        residue_features = []
        
        # Process first model and first chain (usually chain A)
        model = next(structure.get_models())
        
        # Find the main protein chain (usually the longest)
        main_chain = None
        max_residues = 0
        
        for chain in model:
            residue_count = sum(1 for res in chain if is_aa(res, standard=True))
            if residue_count > max_residues:
                max_residues = residue_count
                main_chain = chain
        
        if main_chain is None:
            raise ValueError("No valid protein chain found in PDB file")
        
        # Extract sequence and coordinates
        for residue in main_chain:
            if not is_aa(residue, standard=True):
                continue
            
            # Get residue name
            res_name = residue.get_resname()
            aa_letter = THREE_TO_ONE.get(res_name, "X")
            sequence_chars.append(aa_letter)
            
            # Get CA coordinates
            if "CA" in residue:
                ca_coord = residue["CA"].get_coord()
                coordinates.append(ca_coord)
            else:
                # If no CA, use geometric center
                coords = [atom.get_coord() for atom in residue]
                center = np.mean(coords, axis=0)
                coordinates.append(center)
            
            # Get amino acid properties
            properties = AA_PROPERTIES.get(aa_letter, AA_PROPERTIES["X"])
            residue_features.append(properties)
        
        self.sequence = "".join(sequence_chars)
        self.coordinates = np.array(coordinates, dtype=np.float32)
        self.residue_features = np.array(residue_features, dtype=np.float32)
        
        # Normalize coordinates (center at origin and scale)
        self.coordinates = self._normalize_coordinates(self.coordinates)
        
        print(f"Processed protein: {len(self.sequence)} residues")
        
        return self.sequence, self.coordinates, self.residue_features
    
    def _normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Normalize coordinates to center at origin and unit scale"""
        # Center at origin
        coords = coords - coords.mean(axis=0)
        
        # Scale to unit variance
        std = coords.std()
        if std > 0:
            coords = coords / std
        
        return coords
    
    def get_local_environments(self, radius: float = 10.0) -> List[np.ndarray]:
        """
        Extract local environment features for each residue
        Returns list of environment feature vectors
        """
        if self.coordinates is None:
            raise ValueError("Must process protein first")
        
        n_residues = len(self.coordinates)
        environments = []
        
        for i in range(n_residues):
            # Calculate distances to all other residues
            distances = np.linalg.norm(self.coordinates - self.coordinates[i], axis=1)
            
            # Find neighbors within radius
            neighbors = np.where((distances < radius) & (distances > 0))[0]
            
            # Calculate environment features
            env_features = np.zeros(10)
            
            if len(neighbors) > 0:
                # Average properties of neighbors
                neighbor_props = self.residue_features[neighbors].mean(axis=0)
                env_features[:8] = neighbor_props
                
                # Add density and average distance
                env_features[8] = len(neighbors) / radius
                env_features[9] = distances[neighbors].mean()
            
            environments.append(env_features)
        
        return environments


class PeptideDataProcessor:
    """Process peptide data from CSV with special handling for duplicates"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.data = None
        self.peptide_features = {}
        
    def load_and_process(self, keep_duplicates: bool = True) -> pd.DataFrame:
        """
        Load CSV and process peptide data
        IMPORTANT: Keeping duplicates helps model learn spatial preferences
        """
        # Read CSV
        self.data = pd.read_csv(self.csv_file)
        
        print(f"Loaded {len(self.data)} peptide-binding pairs")
        
        if keep_duplicates:
            # Keep duplicates to learn spatial variations
            print(f"Keeping all duplicates for spatial learning")
            
            # Add occurrence index for duplicates
            self.data['occurrence_idx'] = self.data.groupby('title').cumcount()
            
            # Calculate statistics for duplicate peptides
            duplicate_stats = self.data.groupby('title')['r_psp_MMGBSA_dG_Bind'].agg(['count', 'mean', 'std', 'min', 'max'])
            print(f"\nDuplicate statistics:")
            print(f"Unique peptides: {len(duplicate_stats)}")
            print(f"Max occurrences: {duplicate_stats['count'].max()}")
            print(f"Average occurrences: {duplicate_stats['count'].mean():.2f}")
        
        # Extract peptide features
        self._extract_peptide_features()
        
        return self.data
    
    def _extract_peptide_features(self):
        """Extract features for each peptide"""
        for peptide in self.data['title'].unique():
            features = self._compute_peptide_features(peptide)
            self.peptide_features[peptide] = features
    
    def _compute_peptide_features(self, peptide: str) -> np.ndarray:
        """
        Compute comprehensive features for a peptide sequence
        """
        features = []
        
        # Sequence-based features
        features.append(len(peptide))  # Length
        
        # Amino acid composition
        aa_counts = np.zeros(20)
        for aa in peptide:
            if aa in AA_TO_INDEX:
                idx = AA_TO_INDEX[aa] - 1
                if idx < 20:
                    aa_counts[idx] += 1
        aa_composition = aa_counts / max(len(peptide), 1)
        features.extend(aa_composition)
        
        # Average properties
        properties = []
        for aa in peptide:
            if aa in AA_PROPERTIES:
                properties.append(AA_PROPERTIES[aa])
        
        if properties:
            avg_props = np.mean(properties, axis=0)
            features.extend(avg_props)
        else:
            features.extend([0] * 8)
        
        # Positional features (N-term, C-term properties)
        if len(peptide) > 0:
            n_term = AA_PROPERTIES.get(peptide[0], AA_PROPERTIES["X"])
            c_term = AA_PROPERTIES.get(peptide[-1], AA_PROPERTIES["X"])
            features.extend(n_term)
            features.extend(c_term)
        else:
            features.extend([0] * 16)
        
        return np.array(features, dtype=np.float32)
    
    def tokenize_peptide(self, peptide: str, max_len: int = 10) -> torch.Tensor:
        """
        Tokenize peptide sequence with padding
        """
        tokens = [AA_TO_INDEX.get(aa, 21) for aa in peptide[:max_len]]
        
        # Pad if necessary
        if len(tokens) < max_len:
            tokens.extend([0] * (max_len - len(tokens)))
        
        return torch.tensor(tokens, dtype=torch.long)


class SpatialFeatureExtractor:
    """Extract spatial features for peptide-protein interactions"""
    
    def __init__(self, protein_coords: np.ndarray, protein_features: np.ndarray):
        self.protein_coords = protein_coords
        self.protein_features = protein_features
        self.n_residues = len(protein_coords)
        
    def get_binding_site_candidates(self, n_sites: int = 10, 
                                   pocket_radius: float = 15.0) -> List[Dict]:
        """
        Identify potential binding sites based on surface accessibility and pocket detection
        """
        binding_sites = []
        
        # Simple grid-based approach to find pockets
        # In practice, you might use more sophisticated methods
        
        # Sample points across protein surface
        min_coord = self.protein_coords.min(axis=0) - pocket_radius
        max_coord = self.protein_coords.max(axis=0) + pocket_radius
        
        # Create grid
        grid_points = []
        grid_size = 5
        for x in np.linspace(min_coord[0], max_coord[0], grid_size):
            for y in np.linspace(min_coord[1], max_coord[1], grid_size):
                for z in np.linspace(min_coord[2], max_coord[2], grid_size):
                    grid_points.append([x, y, z])
        
        grid_points = np.array(grid_points)
        
        # Evaluate each grid point
        for point in grid_points:
            distances = np.linalg.norm(self.protein_coords - point, axis=1)
            nearby_residues = np.where(distances < pocket_radius)[0]
            
            if len(nearby_residues) >= 5:  # At least 5 residues nearby
                # Calculate pocket depth (how enclosed the point is)
                enclosure = np.sum(distances[nearby_residues] < pocket_radius/2) / len(nearby_residues)
                
                # Calculate feature diversity in pocket
                if len(nearby_residues) > 0:
                    pocket_features = self.protein_features[nearby_residues].mean(axis=0)
                    
                    binding_sites.append({
                        'center': point,
                        'residues': nearby_residues,
                        'enclosure': enclosure,
                        'features': pocket_features,
                        'size': len(nearby_residues)
                    })
        
        # Sort by enclosure score and return top N
        binding_sites.sort(key=lambda x: x['enclosure'], reverse=True)
        return binding_sites[:n_sites]
    
    def compute_interaction_features(self, peptide_features: np.ndarray, 
                                    binding_site: Dict) -> np.ndarray:
        """
        Compute interaction features between peptide and binding site
        """
        features = []
        
        # Electrostatic complementarity
        site_charge = binding_site['features'][2]  # Charge index
        pep_charge = peptide_features[2] if len(peptide_features) > 2 else 0
        features.append(-site_charge * pep_charge)  # Opposite charges attract
        
        # Hydrophobic matching
        site_hydrophob = binding_site['features'][0]
        pep_hydrophob = peptide_features[0] if len(peptide_features) > 0 else 0
        features.append(site_hydrophob * pep_hydrophob)
        
        # Size compatibility
        site_volume = binding_site['size'] * 100  # Approximate pocket volume
        pep_volume = peptide_features[1] if len(peptide_features) > 1 else 100
        features.append(-abs(site_volume - pep_volume) / 100)
        
        # H-bond potential
        site_donor = binding_site['features'][5] if len(binding_site['features']) > 5 else 0
        site_acceptor = binding_site['features'][6] if len(binding_site['features']) > 6 else 0
        pep_donor = peptide_features[5] if len(peptide_features) > 5 else 0
        pep_acceptor = peptide_features[6] if len(peptide_features) > 6 else 0
        features.append(site_donor * pep_acceptor + site_acceptor * pep_donor)
        
        # Aromatic stacking
        site_aromatic = binding_site['features'][4] if len(binding_site['features']) > 4 else 0
        pep_aromatic = peptide_features[4] if len(peptide_features) > 4 else 0
        features.append(site_aromatic * pep_aromatic)
        
        # Pocket enclosure score
        features.append(binding_site['enclosure'])
        
        return np.array(features, dtype=np.float32)


def prepare_training_data(pdb_file: str, csv_file: str, max_peptide_len: int = 10):
    """
    Prepare complete training dataset
    """
    print("Processing protein structure...")
    protein_processor = ProteinStructureProcessor(pdb_file)
    protein_seq, protein_coords, protein_features = protein_processor.process_protein()
    
    print("\nProcessing peptide data...")
    peptide_processor = PeptideDataProcessor(csv_file)
    peptide_data = peptide_processor.load_and_process(keep_duplicates=True)
    
    print("\nExtracting spatial features...")
    spatial_extractor = SpatialFeatureExtractor(protein_coords, protein_features)
    binding_sites = spatial_extractor.get_binding_site_candidates()
    
    # Prepare training samples
    training_samples = []
    
    for idx, row in peptide_data.iterrows():
        peptide = row['title']
        delta_g = row['r_psp_MMGBSA_dG_Bind']
        occurrence_idx = row.get('occurrence_idx', 0)
        
        # Tokenize peptide
        peptide_tokens = peptide_processor.tokenize_peptide(peptide, max_peptide_len)
        
        # Get peptide features
        peptide_features = peptide_processor.peptide_features[peptide]
        
        # For duplicates, use different binding sites
        site_idx = occurrence_idx % len(binding_sites)
        binding_site = binding_sites[site_idx]
        
        # Compute interaction features
        interaction_features = spatial_extractor.compute_interaction_features(
            peptide_features, binding_site
        )
        
        sample = {
            'peptide_seq': peptide,
            'peptide_tokens': peptide_tokens,
            'peptide_features': peptide_features,
            'protein_seq': protein_seq,
            'protein_coords': protein_coords,
            'protein_features': protein_features,
            'binding_site': binding_site,
            'interaction_features': interaction_features,
            'delta_g': delta_g,
            'occurrence_idx': occurrence_idx
        }
        
        training_samples.append(sample)
    
    print(f"\nPrepared {len(training_samples)} training samples")
    
    # Normalize features
    scaler = StandardScaler()
    all_peptide_features = np.stack([s['peptide_features'] for s in training_samples])
    all_peptide_features = scaler.fit_transform(all_peptide_features)
    
    for i, sample in enumerate(training_samples):
        sample['peptide_features'] = all_peptide_features[i]
    
    return training_samples, protein_processor, peptide_processor, spatial_extractor


if __name__ == "__main__":
    # Test the data processing
    samples, prot_proc, pep_proc, spatial_ext = prepare_training_data(
        "PSMA.pdb", 
        "PSMA_MMGBSA.csv"
    )
    
    print(f"\nSample data structure:")
    sample = samples[0]
    for key, value in sample.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            print(f"  {key}: shape {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
        else:
            print(f"  {key}: {type(value).__name__}")
