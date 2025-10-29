"""
Generate a comprehensive architecture diagram showing complete data flow
through the peptide binding prediction model
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def create_architecture_diagram():
    # Create figure with high DPI for clarity
    fig, ax = plt.subplots(1, 1, figsize=(20, 24), dpi=150)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 30)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E8F4FD',
        'preprocessing': '#B8E6B8',
        'embedding': '#FFE4B5',
        'attention': '#FFB6C1',
        'se3': '#DDA0DD',
        'fusion': '#F0E68C',
        'output': '#87CEEB',
        'loss': '#FFA07A'
    }
    
    # Title
    ax.text(10, 29, 'Peptide Binding Prediction Model: Complete Data Flow', 
            fontsize=18, fontweight='bold', ha='center')
    ax.text(10, 28.3, 'From Raw Input to ΔG Prediction', 
            fontsize=14, ha='center', style='italic')
    
    y_pos = 27
    
    # ============ SECTION 1: RAW INPUT DATA ============
    ax.text(10, y_pos, '1. RAW INPUT DATA', fontsize=14, fontweight='bold', ha='center')
    y_pos -= 0.8
    
    # PDB File
    pdb_box = FancyBboxPatch((1, y_pos-1.2), 3.5, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(pdb_box)
    ax.text(2.75, y_pos-0.7, 'PSMA.pdb', fontsize=11, fontweight='bold', ha='center')
    ax.text(2.75, y_pos-1, '692 residues\n3D coordinates', fontsize=9, ha='center')
    
    # CSV File
    csv_box = FancyBboxPatch((5.5, y_pos-1.2), 3.5, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(csv_box)
    ax.text(7.25, y_pos-0.7, 'PSMA_MMGBSA.csv', fontsize=11, fontweight='bold', ha='center')
    ax.text(7.25, y_pos-1, '40,009 samples\n9,526 peptides', fontsize=9, ha='center')
    
    # Duplicate info
    dup_box = FancyBboxPatch((11, y_pos-1.2), 8, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#FFFACD', edgecolor='orange', linewidth=2, linestyle='--')
    ax.add_patch(dup_box)
    ax.text(15, y_pos-0.6, '⚡ KEY INNOVATION: Duplicate Handling', fontsize=10, fontweight='bold', ha='center')
    ax.text(15, y_pos-0.95, 'Same peptide → Multiple spatial orientations → Learn binding preferences', 
            fontsize=9, ha='center', style='italic')
    
    y_pos -= 2
    
    # ============ SECTION 2: DATA PROCESSING ============
    ax.text(10, y_pos, '2. DATA PREPROCESSING', fontsize=14, fontweight='bold', ha='center')
    y_pos -= 0.8
    
    # Protein Processing
    prot_proc = FancyBboxPatch((0.5, y_pos-1.5), 4, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['preprocessing'], edgecolor='darkgreen', linewidth=2)
    ax.add_patch(prot_proc)
    ax.text(2.5, y_pos-0.5, 'Protein Processing', fontsize=11, fontweight='bold', ha='center')
    ax.text(2.5, y_pos-0.8, '• Extract Cα coordinates', fontsize=9, ha='center')
    ax.text(2.5, y_pos-1.05, '• Normalize (center, scale)', fontsize=9, ha='center')
    ax.text(2.5, y_pos-1.3, '• Compute local environments', fontsize=9, ha='center')
    
    # Peptide Processing
    pep_proc = FancyBboxPatch((5, y_pos-1.5), 4, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['preprocessing'], edgecolor='darkgreen', linewidth=2)
    ax.add_patch(pep_proc)
    ax.text(7, y_pos-0.5, 'Peptide Processing', fontsize=11, fontweight='bold', ha='center')
    ax.text(7, y_pos-0.8, '• Tokenize sequences', fontsize=9, ha='center')
    ax.text(7, y_pos-1.05, '• Extract 45 features', fontsize=9, ha='center')
    ax.text(7, y_pos-1.3, '• Keep occurrence index', fontsize=9, ha='center')
    
    # Spatial Features
    spatial_proc = FancyBboxPatch((9.5, y_pos-1.5), 5, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['preprocessing'], edgecolor='darkgreen', linewidth=2)
    ax.add_patch(spatial_proc)
    ax.text(12, y_pos-0.5, 'Spatial Feature Extraction', fontsize=11, fontweight='bold', ha='center')
    ax.text(12, y_pos-0.8, '• Detect binding sites (10 sites)', fontsize=9, ha='center')
    ax.text(12, y_pos-1.05, '• Calculate pocket properties', fontsize=9, ha='center')
    ax.text(12, y_pos-1.3, '• Compute 6 interaction features', fontsize=9, ha='center')
    
    # Feature Details
    feat_detail = FancyBboxPatch((15, y_pos-1.5), 4.5, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#F0F8FF', edgecolor='blue', linewidth=1)
    ax.add_patch(feat_detail)
    ax.text(17.25, y_pos-0.5, '8 AA Properties:', fontsize=10, fontweight='bold', ha='center')
    ax.text(17.25, y_pos-0.75, 'Hydrophobicity, Volume', fontsize=8, ha='center')
    ax.text(17.25, y_pos-0.95, 'Charge, Polarity, Aromatic', fontsize=8, ha='center')
    ax.text(17.25, y_pos-1.15, 'H-bond D/A, Flexibility', fontsize=8, ha='center')
    
    y_pos -= 2.5
    
    # ============ SECTION 3: TOKENIZATION & EMBEDDING ============
    ax.text(10, y_pos, '3. TOKENIZATION & EMBEDDING', fontsize=14, fontweight='bold', ha='center')
    y_pos -= 0.8
    
    # Peptide Tokenization
    pep_token = FancyBboxPatch((1, y_pos-1.2), 4, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['embedding'], edgecolor='darkorange', linewidth=2)
    ax.add_patch(pep_token)
    ax.text(3, y_pos-0.4, 'Peptide Tokenization', fontsize=11, fontweight='bold', ha='center')
    ax.text(3, y_pos-0.65, 'DHRVYR → [3,7,15,18,20,15]', fontsize=9, ha='center', family='monospace')
    ax.text(3, y_pos-0.9, 'Pad to length 10', fontsize=9, ha='center')
    
    # Protein Tokenization
    prot_token = FancyBboxPatch((5.5, y_pos-1.2), 4, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['embedding'], edgecolor='darkorange', linewidth=2)
    ax.add_patch(prot_token)
    ax.text(7.5, y_pos-0.4, 'Protein Tokenization', fontsize=11, fontweight='bold', ha='center')
    ax.text(7.5, y_pos-0.65, '692 AA → [1,15,12,...]', fontsize=9, ha='center', family='monospace')
    ax.text(7.5, y_pos-0.9, 'Full sequence encoding', fontsize=9, ha='center')
    
    # Property Embedding
    prop_embed = FancyBboxPatch((10, y_pos-1.2), 5, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['embedding'], edgecolor='darkorange', linewidth=2)
    ax.add_patch(prop_embed)
    ax.text(12.5, y_pos-0.4, 'Property-Aware Embedding', fontsize=11, fontweight='bold', ha='center')
    ax.text(12.5, y_pos-0.65, 'Token Embed (120D) + Properties (8D)', fontsize=9, ha='center')
    ax.text(12.5, y_pos-0.9, '= 128D embedding per AA', fontsize=9, ha='center')
    
    # 3D Coords
    coords_box = FancyBboxPatch((15.5, y_pos-1.2), 4, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['embedding'], edgecolor='darkorange', linewidth=2)
    ax.add_patch(coords_box)
    ax.text(17.5, y_pos-0.4, '3D Coordinates', fontsize=11, fontweight='bold', ha='center')
    ax.text(17.5, y_pos-0.65, '[batch, 692, 3]', fontsize=9, ha='center', family='monospace')
    ax.text(17.5, y_pos-0.9, 'Normalized xyz coords', fontsize=9, ha='center')
    
    y_pos -= 2
    
    # ============ SECTION 4: ENCODER LAYERS ============
    ax.text(10, y_pos, '4. ENCODER NETWORKS', fontsize=14, fontweight='bold', ha='center')
    y_pos -= 0.8
    
    # Peptide Encoder
    pep_enc = FancyBboxPatch((0.5, y_pos-2), 6, 2, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['attention'], edgecolor='darkred', linewidth=2)
    ax.add_patch(pep_enc)
    ax.text(3.5, y_pos-0.3, 'PEPTIDE ENCODER', fontsize=11, fontweight='bold', ha='center')
    ax.text(3.5, y_pos-0.6, '1. Embedding projection → 256D', fontsize=9, ha='center')
    ax.text(3.5, y_pos-0.85, '2. Positional encoding (sin/cos)', fontsize=9, ha='center')
    ax.text(3.5, y_pos-1.1, '3. Self-attention (8 heads)', fontsize=9, ha='center')
    ax.text(3.5, y_pos-1.35, '4. Layer norm + dropout', fontsize=9, ha='center')
    ax.text(3.5, y_pos-1.6, '5. Feed-forward (256→1024→256)', fontsize=9, ha='center')
    ax.text(3.5, y_pos-1.85, 'Output: [batch, 10, 256]', fontsize=9, ha='center', fontweight='bold')
    
    # Protein Encoder with SE3
    prot_enc = FancyBboxPatch((7, y_pos-2), 6, 2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['se3'], edgecolor='purple', linewidth=2)
    ax.add_patch(prot_enc)
    ax.text(10, y_pos-0.3, 'PROTEIN ENCODER + SE(3)', fontsize=11, fontweight='bold', ha='center')
    ax.text(10, y_pos-0.6, '1. Sequence encoding (like peptide)', fontsize=9, ha='center')
    ax.text(10, y_pos-0.85, '2. SE(3) equivariant layers (×3):', fontsize=9, ha='center')
    ax.text(10, y_pos-1.1, '   • Scalar net: features→256D', fontsize=8, ha='center')
    ax.text(10, y_pos-1.3, '   • Vector net: coords refinement', fontsize=8, ha='center')
    ax.text(10, y_pos-1.5, '   • Coordinate update: Δ×0.1', fontsize=8, ha='center')
    ax.text(10, y_pos-1.75, 'Output: [batch, 692, 256] + coords', fontsize=9, ha='center', fontweight='bold')
    
    # SE3 Detail Box
    se3_detail = FancyBboxPatch((13.5, y_pos-2), 6, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E6E6FA', edgecolor='indigo', linewidth=2)
    ax.add_patch(se3_detail)
    ax.text(16.5, y_pos-0.3, 'SE(3) EQUIVARIANCE', fontsize=11, fontweight='bold', ha='center')
    ax.text(16.5, y_pos-0.6, '⚡ Preserves 3D geometry', fontsize=9, ha='center')
    ax.text(16.5, y_pos-0.85, 'Rotation invariant', fontsize=9, ha='center')
    ax.text(16.5, y_pos-1.1, 'Translation invariant', fontsize=9, ha='center')
    ax.text(16.5, y_pos-1.35, 'Learns spatial patterns', fontsize=9, ha='center')
    ax.text(16.5, y_pos-1.6, 'Refines coordinates iteratively', fontsize=9, ha='center')
    
    y_pos -= 2.8
    
    # ============ SECTION 5: CROSS-ATTENTION ============
    ax.text(10, y_pos, '5. CROSS-ATTENTION LAYERS (×4)', fontsize=14, fontweight='bold', ha='center')
    y_pos -= 0.8
    
    # Bidirectional attention
    cross_attn = FancyBboxPatch((2, y_pos-2.2), 16, 2.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['attention'], edgecolor='red', linewidth=3)
    ax.add_patch(cross_attn)
    
    # Left side - Peptide attends to Protein
    ax.text(6, y_pos-0.3, 'Peptide → Protein Attention', fontsize=11, fontweight='bold', ha='center')
    ax.text(6, y_pos-0.6, 'Query: Peptide features', fontsize=9, ha='center')
    ax.text(6, y_pos-0.85, 'Key/Value: Protein features', fontsize=9, ha='center')
    ax.text(6, y_pos-1.1, '8 attention heads', fontsize=9, ha='center')
    ax.text(6, y_pos-1.35, 'Learn: Which protein regions', fontsize=9, ha='center', style='italic')
    ax.text(6, y_pos-1.55, 'interact with this peptide?', fontsize=9, ha='center', style='italic')
    
    # Middle - Bidirectional arrow
    arrow1 = FancyArrowPatch((8.5, y_pos-1), (11.5, y_pos-1),
                            connectionstyle="arc3,rad=0", 
                            arrowstyle='<->', mutation_scale=25, linewidth=3, color='darkred')
    ax.add_patch(arrow1)
    ax.text(10, y_pos-0.7, '⟷', fontsize=20, ha='center', color='darkred')
    ax.text(10, y_pos-1.4, 'Bidirectional', fontsize=10, ha='center', fontweight='bold')
    ax.text(10, y_pos-1.6, 'Information Flow', fontsize=10, ha='center', fontweight='bold')
    
    # Right side - Protein attends to Peptide
    ax.text(14, y_pos-0.3, 'Protein → Peptide Attention', fontsize=11, fontweight='bold', ha='center')
    ax.text(14, y_pos-0.6, 'Query: Protein features', fontsize=9, ha='center')
    ax.text(14, y_pos-0.85, 'Key/Value: Peptide features', fontsize=9, ha='center')
    ax.text(14, y_pos-1.1, '8 attention heads', fontsize=9, ha='center')
    ax.text(14, y_pos-1.35, 'Learn: Which peptide positions', fontsize=9, ha='center', style='italic')
    ax.text(14, y_pos-1.55, 'are important for binding?', fontsize=9, ha='center', style='italic')
    
    # Repeat indicator
    ax.text(10, y_pos-1.95, '↻ Repeat 4 times for deep interaction learning', fontsize=10, ha='center', style='italic')
    
    y_pos -= 3
    
    # ============ SECTION 6: BINDING SITE ATTENTION ============
    ax.text(10, y_pos, '6. BINDING SITE ATTENTION', fontsize=14, fontweight='bold', ha='center')
    y_pos -= 0.8
    
    binding_attn = FancyBboxPatch((3, y_pos-1.5), 14, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#FFEFD5', edgecolor='brown', linewidth=2)
    ax.add_patch(binding_attn)
    ax.text(10, y_pos-0.3, 'Learned Spatial Focus', fontsize=11, fontweight='bold', ha='center')
    ax.text(10, y_pos-0.6, 'Input: Protein features [batch, 692, 256]', fontsize=9, ha='center')
    ax.text(10, y_pos-0.85, 'Attention weights: σ(Linear(256→64→1))', fontsize=9, ha='center', family='monospace')
    ax.text(10, y_pos-1.1, 'Weighted aggregation → Focus on binding pockets', fontsize=9, ha='center')
    ax.text(10, y_pos-1.35, '⚡ Learns from duplicates: different sites = different ΔG', fontsize=9, ha='center', style='italic')
    
    y_pos -= 2.2
    
    # ============ SECTION 7: FEATURE AGGREGATION ============
    ax.text(10, y_pos, '7. GLOBAL FEATURE AGGREGATION', fontsize=14, fontweight='bold', ha='center')
    y_pos -= 0.8
    
    # Feature boxes
    pep_global = FancyBboxPatch((1, y_pos-1), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['fusion'], edgecolor='olive', linewidth=2)
    ax.add_patch(pep_global)
    ax.text(2.5, y_pos-0.3, 'Peptide Global', fontsize=10, fontweight='bold', ha='center')
    ax.text(2.5, y_pos-0.55, 'Mean pooling', fontsize=9, ha='center')
    ax.text(2.5, y_pos-0.8, '[batch, 256]', fontsize=9, ha='center', family='monospace')
    
    prot_global = FancyBboxPatch((4.5, y_pos-1), 3, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['fusion'], edgecolor='olive', linewidth=2)
    ax.add_patch(prot_global)
    ax.text(6, y_pos-0.3, 'Protein Global', fontsize=10, fontweight='bold', ha='center')
    ax.text(6, y_pos-0.55, 'Weighted pooling', fontsize=9, ha='center')
    ax.text(6, y_pos-0.8, '[batch, 256]', fontsize=9, ha='center', family='monospace')
    
    interaction = FancyBboxPatch((8, y_pos-1), 3.5, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['fusion'], edgecolor='olive', linewidth=2)
    ax.add_patch(interaction)
    ax.text(9.75, y_pos-0.3, 'Interaction', fontsize=10, fontweight='bold', ha='center')
    ax.text(9.75, y_pos-0.55, 'Pep × Prot', fontsize=9, ha='center')
    ax.text(9.75, y_pos-0.8, '[batch, 256]', fontsize=9, ha='center', family='monospace')
    
    difference = FancyBboxPatch((12, y_pos-1), 3.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['fusion'], edgecolor='olive', linewidth=2)
    ax.add_patch(difference)
    ax.text(13.75, y_pos-0.3, 'Difference', fontsize=10, fontweight='bold', ha='center')
    ax.text(13.75, y_pos-0.55, '|Pep - Prot|', fontsize=9, ha='center')
    ax.text(13.75, y_pos-0.8, '[batch, 256]', fontsize=9, ha='center', family='monospace')
    
    spatial_feat = FancyBboxPatch((16, y_pos-1), 3.5, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['fusion'], edgecolor='olive', linewidth=2)
    ax.add_patch(spatial_feat)
    ax.text(17.75, y_pos-0.3, 'Spatial Features', fontsize=10, fontweight='bold', ha='center')
    ax.text(17.75, y_pos-0.55, '6 interact + 45 pep', fontsize=9, ha='center')
    ax.text(17.75, y_pos-0.8, '[batch, 256]', fontsize=9, ha='center', family='monospace')
    
    y_pos -= 1.8
    
    # ============ SECTION 8: GLOBAL INTERACTION ============
    ax.text(10, y_pos, '8. GLOBAL INTERACTION PROCESSING', fontsize=14, fontweight='bold', ha='center')
    y_pos -= 0.8
    
    global_proc = FancyBboxPatch((5, y_pos-1.5), 10, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#FFE4E1', edgecolor='crimson', linewidth=2)
    ax.add_patch(global_proc)
    ax.text(10, y_pos-0.3, 'Deep Feature Integration', fontsize=11, fontweight='bold', ha='center')
    ax.text(10, y_pos-0.6, 'Concatenate all: [Pep, Prot, Pep×Prot, |Pep-Prot|] → 1024D', fontsize=9, ha='center')
    ax.text(10, y_pos-0.85, 'Linear(1024 → 512) + LayerNorm + ReLU + Dropout', fontsize=9, ha='center', family='monospace')
    ax.text(10, y_pos-1.1, 'Linear(512 → 256) → Compressed representation', fontsize=9, ha='center', family='monospace')
    ax.text(10, y_pos-1.35, 'Output: [batch, 256] interaction features', fontsize=9, ha='center', fontweight='bold')
    
    y_pos -= 2.3
    
    # ============ SECTION 9: PREDICTION HEAD ============
    ax.text(10, y_pos, '9. PREDICTION HEAD', fontsize=14, fontweight='bold', ha='center')
    y_pos -= 0.8
    
    pred_head = FancyBboxPatch((4, y_pos-1.8), 12, 1.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], edgecolor='darkblue', linewidth=3)
    ax.add_patch(pred_head)
    ax.text(10, y_pos-0.3, 'Final ΔG Prediction Network', fontsize=11, fontweight='bold', ha='center')
    ax.text(10, y_pos-0.6, 'Input: [Interaction (256) + Spatial (256)] = 512D', fontsize=9, ha='center')
    ax.text(10, y_pos-0.85, 'Layer 1: Linear(512→256) + LayerNorm + ReLU + Dropout(0.2)', fontsize=9, ha='center', family='monospace')
    ax.text(10, y_pos-1.1, 'Layer 2: Linear(256→128) + LayerNorm + ReLU + Dropout(0.1)', fontsize=9, ha='center', family='monospace')
    ax.text(10, y_pos-1.35, 'Layer 3: Linear(128→64) + ReLU', fontsize=9, ha='center', family='monospace')
    ax.text(10, y_pos-1.6, 'Output: Linear(64→1) → ΔG (kcal/mol)', fontsize=10, ha='center', fontweight='bold')
    
    y_pos -= 2.5
    
    # ============ SECTION 10: LOSS & OPTIMIZATION ============
    ax.text(10, y_pos, '10. LOSS COMPUTATION & BACKPROPAGATION', fontsize=14, fontweight='bold', ha='center')
    y_pos -= 0.8
    
    loss_box = FancyBboxPatch((2, y_pos-1.8), 16, 1.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['loss'], edgecolor='red', linewidth=3)
    ax.add_patch(loss_box)
    ax.text(10, y_pos-0.3, 'Dynamic Multi-Component Loss', fontsize=11, fontweight='bold', ha='center')
    ax.text(5, y_pos-0.6, 'Early (epoch<10):', fontsize=9, ha='center', fontweight='bold')
    ax.text(5, y_pos-0.8, 'MSE + 0.1×MAE', fontsize=8, ha='center', family='monospace')
    
    ax.text(10, y_pos-0.6, 'Mid (10≤epoch<30):', fontsize=9, ha='center', fontweight='bold')
    ax.text(10, y_pos-0.8, '0.5×MSE + 0.3×MAE +', fontsize=8, ha='center', family='monospace')
    ax.text(10, y_pos-0.95, '0.1×Huber + 0.1×Corr', fontsize=8, ha='center', family='monospace')
    
    ax.text(15, y_pos-0.6, 'Late (epoch≥30):', fontsize=9, ha='center', fontweight='bold')
    ax.text(15, y_pos-0.8, '0.3×MSE + 0.2×MAE +', fontsize=8, ha='center', family='monospace')
    ax.text(15, y_pos-0.95, '0.2×Huber + 0.3×Corr', fontsize=8, ha='center', family='monospace')
    
    ax.text(10, y_pos-1.25, 'Optimizer: AdamW (lr=5e-4, weight_decay=1e-5)', fontsize=9, ha='center', fontweight='bold')
    ax.text(10, y_pos-1.5, 'Gradient clipping: max_norm=1.0', fontsize=9, ha='center')
    ax.text(10, y_pos-1.7, 'Scheduler: ReduceLROnPlateau + 5-epoch warmup', fontsize=9, ha='center')
    
    y_pos -= 2.5
    
    # ============ FINAL OUTPUT ============
    output_box = FancyBboxPatch((7, y_pos-0.8), 6, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='gold', edgecolor='black', linewidth=3)
    ax.add_patch(output_box)
    ax.text(10, y_pos-0.4, 'OUTPUT: ΔG Binding Affinity', fontsize=12, fontweight='bold', ha='center')
    ax.text(10, y_pos-0.65, 'More negative = Stronger binding', fontsize=10, ha='center', style='italic')
    
    # ============ ARROWS SHOWING DATA FLOW ============
    # Add some key flow arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='gray', alpha=0.6)
    
    # Input to processing
    ax.annotate('', xy=(2.75, 23.3), xytext=(2.75, 24.3), arrowprops=arrow_props)
    ax.annotate('', xy=(7.25, 23.3), xytext=(7.25, 24.3), arrowprops=arrow_props)
    
    # Processing to tokenization
    ax.annotate('', xy=(3, 21.5), xytext=(2.5, 22.5), arrowprops=arrow_props)
    ax.annotate('', xy=(7.5, 21.5), xytext=(7, 22.5), arrowprops=arrow_props)
    
    # Tokenization to encoders
    ax.annotate('', xy=(3.5, 18.7), xytext=(3, 20.3), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 18.7), xytext=(7.5, 20.3), arrowprops=arrow_props)
    
    # Encoders to attention
    ax.annotate('', xy=(10, 15.9), xytext=(10, 16.7), arrowprops=arrow_props)
    
    # Through all remaining stages
    ax.annotate('', xy=(10, 12.9), xytext=(10, 13.7), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 10.7), xytext=(10, 11.4), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 8.9), xytext=(10, 9.6), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 6.6), xytext=(10, 7.4), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 4.8), xytext=(10, 5.6), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 2.2), xytext=(10, 3.0), arrowprops=arrow_props)
    
    # Add parameter count boxes
    param_box = FancyBboxPatch((0.2, 0.2), 3.5, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor='lightgray', edgecolor='black', linewidth=1)
    ax.add_patch(param_box)
    ax.text(1.95, 0.6, 'Total Parameters:', fontsize=9, fontweight='bold', ha='center')
    ax.text(1.95, 0.35, '~7.8M', fontsize=10, ha='center', fontweight='bold')
    
    # Add batch flow indicator
    batch_box = FancyBboxPatch((16.3, 0.2), 3.5, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor='lightgray', edgecolor='black', linewidth=1)
    ax.add_patch(batch_box)
    ax.text(18.05, 0.6, 'Batch Processing:', fontsize=9, fontweight='bold', ha='center')
    ax.text(18.05, 0.35, 'batch_size=32', fontsize=10, ha='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Generate and save the diagram
if __name__ == "__main__":
    print("Generating comprehensive architecture diagram...")
    fig = create_architecture_diagram()
    
    # Save in multiple formats for best quality
    fig.savefig('model_architecture_complete.png', dpi=300, bbox_inches='tight')
    fig.savefig('model_architecture_complete.pdf', bbox_inches='tight')
    
    print("Saved as: model_architecture_complete.png")
    print("Saved as: model_architecture_complete.pdf")
    
    # Also create a simplified version
    print("\nGenerating simplified flow diagram...")
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10), dpi=150)
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Title
    ax2.text(7, 9.5, 'Peptide Binding Model: Simplified Flow', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Main flow boxes
    boxes = [
        (7, 8.5, 'INPUT\nPeptide: DHRVYR\nProtein: PSMA (692 AA)', '#E8F4FD'),
        (7, 7.2, 'TOKENIZATION\nPeptide → [3,7,15,18,20,15]\n+ 8 Properties/AA', '#FFE4B5'),
        (7, 5.9, 'ENCODING\nPeptide Encoder + Protein SE(3)\n3D structure aware', '#DDA0DD'),
        (7, 4.6, 'CROSS-ATTENTION\n4 layers × 8 heads\nBidirectional interaction', '#FFB6C1'),
        (7, 3.3, 'FEATURE FUSION\nGlobal aggregation\nSpatial features', '#F0E68C'),
        (7, 2.0, 'PREDICTION\nDeep network\n512→256→128→64→1', '#87CEEB'),
        (7, 0.7, 'OUTPUT\nΔG = -106.06 kcal/mol', 'gold'),
    ]
    
    for x, y, text, color in boxes:
        box = FancyBboxPatch((x-2, y-0.4), 4, 0.7, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color, edgecolor='black', linewidth=2)
        ax2.add_patch(box)
        lines = text.split('\n')
        ax2.text(x, y, lines[0], fontsize=11, fontweight='bold', ha='center')
        for i, line in enumerate(lines[1:], 1):
            ax2.text(x, y-0.2*i, line, fontsize=9, ha='center')
    
    # Flow arrows
    for i in range(len(boxes)-1):
        ax2.annotate('', xy=(7, boxes[i+1][1]+0.3), xytext=(7, boxes[i][1]-0.4),
                    arrowprops=dict(arrowstyle='->', lw=3, color='darkblue'))
    
    # Key innovations on the side
    innovations = [
        (1.5, 6, '⚡ INNOVATIONS', 'orange'),
        (1.5, 5.5, '• Duplicate handling\n  for spatial learning', 'black'),
        (1.5, 4.8, '• SE(3) equivariance\n  for 3D structure', 'black'),
        (1.5, 4.1, '• Property-aware\n  tokenization', 'black'),
        (1.5, 3.4, '• Dynamic loss\n  weighting', 'black'),
        (1.5, 2.7, '• Binding site\n  attention', 'black'),
    ]
    
    for x, y, text, color in innovations:
        lines = text.split('\n')
        ax2.text(x, y, lines[0], fontsize=10, fontweight='bold' if color=='orange' else 'normal', 
                color=color, ha='center')
        for i, line in enumerate(lines[1:], 1):
            ax2.text(x, y-0.2*i, line, fontsize=8, ha='center')
    
    # Parameters info
    param_info = [
        (12.5, 6, '📊 MODEL STATS', 'darkgreen'),
        (12.5, 5.5, '7.8M parameters', 'black'),
        (12.5, 5.2, '256D hidden dim', 'black'),
        (12.5, 4.9, '8 attention heads', 'black'),
        (12.5, 4.6, '4 cross-attn layers', 'black'),
        (12.5, 4.3, '3 SE(3) layers', 'black'),
    ]
    
    for x, y, text, color in param_info:
        ax2.text(x, y, text, fontsize=10, fontweight='bold' if color=='darkgreen' else 'normal',
                color=color, ha='center')
    
    plt.tight_layout()
    fig2.savefig('model_architecture_simple.png', dpi=300, bbox_inches='tight')
    print("Saved as: model_architecture_simple.png")
    
    plt.show()
    print("\nArchitecture diagrams generated successfully!")
