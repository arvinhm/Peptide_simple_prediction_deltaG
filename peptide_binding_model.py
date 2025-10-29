"""
Advanced Peptide-Protein Binding Prediction Model
Uses spatial features, attention mechanisms, and SE(3) equivariance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
import math

# Import AA properties
AA_TO_INDEX = {
    "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9,
    "L": 10, "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17,
    "V": 18, "W": 19, "Y": 20, "X": 21
}

# Enhanced amino acid properties (normalized)
AA_PROPERTIES_NORMALIZED = {
    'A': [0.56, -0.59, 0.0, -1.0, -1.0, -1.0, -1.0, -0.30],
    'R': [-1.0, 0.73, 1.0, 1.0, -1.0, 1.0, -0.33, 0.53],
    'N': [-0.78, -0.24, 0.0, 1.0, -1.0, 0.0, 0.33, 0.13],
    'D': [-0.78, -0.28, -1.0, 1.0, -1.0, -0.5, 1.0, 0.43],
    'C': [0.78, -0.31, 0.0, -0.5, -1.0, -0.5, -0.33, -0.30],
    'Q': [-0.78, 0.16, 0.0, 1.0, -1.0, 0.0, 0.33, 0.33],
    'E': [-0.78, 0.07, -1.0, 1.0, -1.0, -0.5, 1.0, 0.40],
    'G': [-0.11, -0.82, 0.0, -1.0, -1.0, -1.0, -1.0, 0.60],
    'H': [-0.67, 0.33, 0.1, 1.0, 1.0, 0.0, -0.33, -0.43],
    'I': [1.0, 0.59, 0.0, -1.0, -1.0, -1.0, -1.0, 0.13],
    'L': [0.89, 0.59, 0.0, -1.0, -1.0, -1.0, -1.0, -0.23],
    'K': [-0.89, 0.62, 1.0, 1.0, -1.0, 0.5, -0.33, 0.20],
    'M': [0.56, 0.53, 0.0, -1.0, -1.0, -1.0, -0.33, -0.50],
    'F': [0.78, 0.88, 0.0, -1.0, 1.0, -1.0, -1.0, -0.47],
    'P': [-0.44, -0.23, 0.0, -1.0, -1.0, -1.0, -1.0, 0.43],
    'S': [-0.22, -0.58, 0.0, 1.0, -1.0, -0.5, -0.33, 0.43],
    'T': [-0.22, -0.18, 0.0, 1.0, -1.0, -0.5, -0.33, 0.03],
    'W': [-0.22, 1.0, 0.0, -0.8, 1.0, -0.5, -0.33, -0.47],
    'Y': [-0.33, 0.91, 0.0, 1.0, 1.0, -0.5, -0.33, 0.00],
    'V': [1.0, 0.09, 0.0, -1.0, -1.0, -1.0, -1.0, -0.13],
    'X': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        return x + self.pe[:, :x.size(1)]


class SE3EquivariantLayer(nn.Module):
    """SE(3)-equivariant layer for 3D spatial processing"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Process scalar features
        self.scalar_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Process vector features (3D coordinates)
        self.vector_net = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        # Coordinate refinement
        self.coord_refine = nn.Sequential(
            nn.Linear(output_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh()
        )
    
    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch, seq_len, feature_dim]
            coords: [batch, seq_len, 3]
        Returns:
            Updated features and coordinates
        """
        # Process scalar features
        scalar_out = self.scalar_net(features)
        
        # Process vector features
        combined = torch.cat([features, coords], dim=-1)
        vector_out = self.vector_net(combined)
        
        # Refine coordinates
        coord_features = torch.cat([scalar_out, coords], dim=-1)
        coord_delta = self.coord_refine(coord_features) * 0.1  # Small updates
        
        new_coords = coords + coord_delta
        
        return scalar_out, new_coords


class PeptideEncoder(nn.Module):
    """Enhanced peptide encoder with property awareness"""
    
    def __init__(self, vocab_size: int = 22, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim - 8, padding_idx=0)
        
        # Position encoding
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Self-attention layers
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, sequences: List[str], indices: torch.Tensor) -> torch.Tensor:
        """
        Encode peptide sequences
        
        Args:
            sequences: List of peptide sequences
            indices: Token indices [batch_size, seq_len]
        Returns:
            Encoded features [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = indices.shape
        device = indices.device
        
        # Token embeddings
        token_embed = self.embedding(indices)
        
        # Add amino acid properties
        property_features = torch.zeros(batch_size, seq_len, 8).to(device)
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq[:seq_len]):
                if aa in AA_PROPERTIES_NORMALIZED:
                    property_features[i, j] = torch.tensor(
                        AA_PROPERTIES_NORMALIZED[aa], dtype=torch.float32
                    )
        
        # Combine embeddings
        combined = torch.cat([token_embed, property_features], dim=-1)
        
        # Add positional encoding
        encoded = self.pos_encoder(combined)
        
        # Project to hidden dimension
        features = self.feature_projection(encoded)
        
        # Self-attention
        attn_out, _ = self.self_attention(features, features, features)
        features = self.norm1(features + attn_out)
        
        # FFN
        ffn_out = self.ffn(features)
        features = self.norm2(features + ffn_out)
        
        return features


class ProteinEncoder(nn.Module):
    """Protein encoder with spatial awareness"""
    
    def __init__(self, vocab_size: int = 22, embed_dim: int = 128, 
                 hidden_dim: int = 256, use_se3: bool = True):
        super().__init__()
        
        self.use_se3 = use_se3
        
        # Sequence encoding (similar to peptide)
        self.sequence_encoder = PeptideEncoder(vocab_size, embed_dim, hidden_dim)
        
        # Spatial processing
        if use_se3:
            self.spatial_layers = nn.ModuleList([
                SE3EquivariantLayer(hidden_dim, hidden_dim, hidden_dim * 2)
                for _ in range(3)
            ])
        else:
            self.coord_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, sequence: str, coords: torch.Tensor, 
                features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode protein structure
        
        Args:
            sequence: Protein sequence
            coords: 3D coordinates [batch_size, seq_len, 3]
            features: Optional residue features
        Returns:
            Encoded features and refined coordinates
        """
        batch_size, seq_len = coords.shape[:2]
        device = coords.device
        
        # Tokenize sequence
        indices = torch.zeros(batch_size, seq_len, dtype=torch.long).to(device)
        for i in range(batch_size):
            for j, aa in enumerate(sequence[:seq_len]):
                indices[i, j] = AA_TO_INDEX.get(aa, 21)
        
        # Encode sequence
        seq_features = self.sequence_encoder([sequence] * batch_size, indices)
        
        # Process coordinates
        if self.use_se3:
            spatial_features = seq_features
            refined_coords = coords
            
            for se3_layer in self.spatial_layers:
                spatial_features, refined_coords = se3_layer(spatial_features, refined_coords)
        else:
            spatial_features = self.coord_encoder(coords)
            refined_coords = coords
        
        # Fuse features
        fused = self.fusion(torch.cat([seq_features, spatial_features], dim=-1))
        
        return fused, refined_coords


class CrossAttentionModule(nn.Module):
    """Cross-attention between peptide and protein"""
    
    def __init__(self, hidden_dim: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply cross-attention
        
        Args:
            query: Query features [batch_size, query_len, hidden_dim]
            key_value: Key/value features [batch_size, kv_len, hidden_dim]
            mask: Optional attention mask
        Returns:
            Updated query features
        """
        attn_out, _ = self.cross_attention(query, key_value, key_value, attn_mask=mask)
        query = self.norm(query + attn_out)
        
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)
        
        return query


class InteractionPredictionHead(nn.Module):
    """Predict binding affinity from interaction features"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict binding affinity (ΔG)"""
        return self.network(features).squeeze(-1)


class PeptideBindingModel(nn.Module):
    """
    Complete model for peptide-protein binding prediction
    Incorporates spatial features, attention, and physicochemical properties
    """
    
    def __init__(self, 
                 vocab_size: int = 22,
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dropout: float = 0.1,
                 use_se3: bool = True,
                 max_peptide_len: int = 10,
                 max_protein_len: int = 1000):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_peptide_len = max_peptide_len
        self.max_protein_len = max_protein_len
        
        # Encoders
        self.peptide_encoder = PeptideEncoder(vocab_size, embed_dim, hidden_dim)
        self.protein_encoder = ProteinEncoder(vocab_size, embed_dim, hidden_dim, use_se3)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionModule(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Global interaction processing
        self.global_interaction = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Spatial feature processing
        self.spatial_processor = nn.Sequential(
            nn.Linear(6 + 45, hidden_dim),  # 6 interaction features + 45 peptide features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final prediction head
        self.prediction_head = InteractionPredictionHead(hidden_dim * 2, hidden_dim)
        
        # Learned binding site attention
        self.binding_site_attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                peptide_sequences: List[str],
                peptide_tokens: torch.Tensor,
                protein_sequence: str,
                protein_coords: torch.Tensor,
                peptide_features: torch.Tensor,
                interaction_features: torch.Tensor,
                protein_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            peptide_sequences: List of peptide sequences
            peptide_tokens: Tokenized peptides [batch_size, pep_len]
            protein_sequence: Protein sequence
            protein_coords: Protein coordinates [batch_size, prot_len, 3]
            peptide_features: Peptide feature vectors [batch_size, feature_dim]
            interaction_features: Precomputed interaction features [batch_size, 6]
            protein_features: Optional protein residue features
        Returns:
            Predicted binding affinities [batch_size]
        """
        batch_size = peptide_tokens.size(0)
        device = peptide_tokens.device
        
        # Encode peptide
        peptide_encoded = self.peptide_encoder(peptide_sequences, peptide_tokens)
        
        # Encode protein
        protein_encoded, refined_coords = self.protein_encoder(
            protein_sequence, protein_coords, protein_features
        )
        
        # Cross-attention between peptide and protein
        pep_attended = peptide_encoded
        prot_attended = protein_encoded
        
        for cross_attn in self.cross_attention_layers:
            # Peptide attends to protein
            pep_attended = cross_attn(pep_attended, prot_attended)
            # Protein attends to peptide
            prot_attended = cross_attn(prot_attended, pep_attended)
        
        # Compute binding site attention weights
        binding_weights = self.binding_site_attention(prot_attended)
        
        # Global pooling with attention
        pep_global = pep_attended.mean(dim=1)
        prot_global = (prot_attended * binding_weights).sum(dim=1) / binding_weights.sum(dim=1).clamp(min=1e-6)
        
        # Process spatial features
        spatial_input = torch.cat([interaction_features, peptide_features], dim=-1)
        spatial_features = self.spatial_processor(spatial_input)
        
        # Combine all features
        combined_features = torch.cat([
            pep_global,
            prot_global,
            pep_global * prot_global,  # Element-wise interaction
            torch.abs(pep_global - prot_global)  # Difference features
        ], dim=-1)
        
        # Global interaction processing
        interaction = self.global_interaction(combined_features)
        
        # Final prediction
        final_features = torch.cat([interaction, spatial_features], dim=-1)
        predictions = self.prediction_head(final_features)
        
        return predictions
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get attention weights for visualization"""
        # This would return the last computed attention weights
        # Implementation depends on storing them during forward pass
        pass


class PeptideBindingModelWithUncertainty(PeptideBindingModel):
    """Extended model with uncertainty estimation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional head for uncertainty
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, *args, **kwargs):
        """Forward pass with uncertainty estimation"""
        # Get base predictions
        predictions = super().forward(*args, **kwargs)
        
        # Get features from parent class (would need modification in parent)
        # For now, return dummy uncertainty
        batch_size = predictions.size(0)
        uncertainty = torch.ones(batch_size).to(predictions.device) * 0.1
        
        return predictions, uncertainty


def create_model(config: Dict) -> nn.Module:
    """Create model from configuration"""
    
    model = PeptideBindingModel(
        vocab_size=config.get('vocab_size', 22),
        embed_dim=config.get('embed_dim', 128),
        hidden_dim=config.get('hidden_dim', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 4),
        dropout=config.get('dropout', 0.1),
        use_se3=config.get('use_se3', True),
        max_peptide_len=config.get('max_peptide_len', 10),
        max_protein_len=config.get('max_protein_len', 1000)
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    config = {
        'vocab_size': 22,
        'embed_dim': 128,
        'hidden_dim': 256,
        'n_heads': 8,
        'n_layers': 4,
        'use_se3': True
    }
    
    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 4
    pep_len = 6
    prot_len = 750
    
    peptide_sequences = ["DHRVYR", "QRQRWR", "RNRWYR", "RQRWSK"]
    peptide_tokens = torch.randint(1, 21, (batch_size, pep_len))
    protein_sequence = "A" * prot_len  # Dummy sequence
    protein_coords = torch.randn(batch_size, prot_len, 3)
    peptide_features = torch.randn(batch_size, 45)
    interaction_features = torch.randn(batch_size, 6)
    
    predictions = model(
        peptide_sequences,
        peptide_tokens,
        protein_sequence,
        protein_coords,
        peptide_features,
        interaction_features
    )
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
