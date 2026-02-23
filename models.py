"""
Architectures de modèles Deep Learning pour classification biologique.

Modèles disponibles:
- MLPClassifier: Perceptron multi-couches avec BatchNorm et Dropout
- TabNetClassifier: Architecture TabNet simplifiée avec attention
- FTTransformer: Transformer pour données tabulaires
- DomainAdversarialNetwork: DANN pour domain adaptation

Auteur: Hadrien Gayap & Assistant IA
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class ResidualBlock(nn.Module):
    """Bloc résiduel avec skip connection."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(x + self.block(x)))


class MLPClassifier(nn.Module):
    """
    Perceptron Multi-Couches avec techniques de régularisation avancées.
    
    Features:
    - Batch Normalization
    - Dropout
    - Skip connections optionnelles
    - GELU activation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128, 64],
        num_classes: int = 2,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        use_residual: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.use_residual = use_residual
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les features avant la couche de classification."""
        return self.encoder(x)


class AttentionBlock(nn.Module):
    """Bloc d'attention pour TabNet."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.bn = nn.BatchNorm1d(output_dim)
    
    def forward(self, x: torch.Tensor, priors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.fc(x)
        x = self.bn(x)
        
        # Masque d'attention avec prior
        mask = torch.softmax(x * priors, dim=-1)
        
        return mask, x


class TabNetClassifier(nn.Module):
    """
    Architecture inspirée de TabNet pour données tabulaires.
    
    Features:
    - Sélection de features par attention
    - Traitement séquentiel des décisions
    - Sparse attention
    
    Reference: https://arxiv.org/abs/1908.07442
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        n_steps: int = 3,
        n_d: int = 64,
        n_a: int = 64,
        gamma: float = 1.5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.n_steps = n_steps
        self.n_d = n_d
        self.n_a = n_a
        self.gamma = gamma
        
        # Batch normalization initiale
        self.initial_bn = nn.BatchNorm1d(input_dim)
        
        # Feature transformer partagé
        self.shared_fc = nn.Linear(input_dim, n_d + n_a)
        self.shared_bn = nn.BatchNorm1d(n_d + n_a)
        
        # Transformers spécifiques à chaque étape
        self.step_fcs = nn.ModuleList([
            nn.Linear(n_a, n_d + n_a) for _ in range(n_steps)
        ])
        self.step_bns = nn.ModuleList([
            nn.BatchNorm1d(n_d + n_a) for _ in range(n_steps)
        ])
        
        # Attention
        self.attention_fcs = nn.ModuleList([
            nn.Linear(n_a, input_dim) for _ in range(n_steps)
        ])
        self.attention_bns = nn.ModuleList([
            nn.BatchNorm1d(input_dim) for _ in range(n_steps)
        ])
        
        # Classifier final
        self.final_fc = nn.Linear(n_d, num_classes)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalisation initiale
        x = self.initial_bn(x)
        
        # Prior scales (initialisé à 1)
        prior_scales = torch.ones(x.shape[0], self.input_dim, device=x.device)
        
        # Agrégation des décisions
        output_aggregate = torch.zeros(x.shape[0], self.n_d, device=x.device)
        
        # Transformer partagé
        shared_out = F.gelu(self.shared_bn(self.shared_fc(x)))
        
        for step in range(self.n_steps):
            # Attention masking
            attention_out = self.attention_bns[step](
                self.attention_fcs[step](shared_out[:, self.n_d:])
            )
            mask = torch.softmax(attention_out * prior_scales, dim=-1)
            
            # Update prior scales
            prior_scales = prior_scales * (self.gamma - mask)
            
            # Masked input
            masked_x = mask * x
            
            # Step-specific transform
            step_out = F.gelu(self.step_bns[step](
                self.step_fcs[step](shared_out[:, self.n_d:])
            ))
            
            # Decision contribution
            decision = F.relu(step_out[:, :self.n_d])
            output_aggregate = output_aggregate + decision
            
            # Update shared output for next step
            shared_out = F.gelu(self.shared_bn(self.shared_fc(masked_x)))
        
        # Classification
        output = self.dropout(output_aggregate)
        return self.final_fc(output)


class PositionalEncoding(nn.Module):
    """Encodage positionnel pour Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer pour données tabulaires.
    
    Chaque feature est traitée comme un token avec son propre embedding.
    
    Reference: https://arxiv.org/abs/2106.11959
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Feature embedding (chaque feature devient un token)
        self.feature_embedding = nn.Linear(1, d_model)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=input_dim + 1, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Tokenize features: (batch, features) -> (batch, features, d_model)
        x = x.unsqueeze(-1)  # (batch, features, 1)
        x = self.feature_embedding(x)  # (batch, features, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, features+1, d_model)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Classification using CLS token
        cls_output = x[:, 0, :]
        return self.classifier(cls_output)


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer pour Domain Adversarial Training."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainAdversarialNetwork(nn.Module):
    """
    Domain Adversarial Neural Network (DANN).
    
    Apprend des features invariantes au domaine pour améliorer
    la généralisation des données synthétiques vers les données réelles.
    
    Reference: https://arxiv.org/abs/1505.07818
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Feature extractor
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_dim = prev_dim
        
        # Label classifier
        self.label_classifier = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.BatchNorm1d(prev_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prev_dim // 2, num_classes)
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.BatchNorm1d(prev_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prev_dim // 2, 1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features
            alpha: Coefficient pour gradient reversal (augmente pendant l'entraînement)
        
        Returns:
            label_output: Prédictions de classe
            domain_output: Prédictions de domaine
        """
        features = self.feature_extractor(x)
        
        # Label prediction
        label_output = self.label_classifier(features)
        
        # Domain prediction with gradient reversal
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reversed_features)
        
        return label_output, domain_output.squeeze(-1)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les features extraites."""
        return self.feature_extractor(x)


class EnsembleModel(nn.Module):
    """
    Modèle d'ensemble combinant plusieurs architectures.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Différents modèles
        self.mlp = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=[256, 128, 64],
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.transformer = FTTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=64,
            n_heads=2,
            n_layers=2,
            dropout=dropout
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_classes * 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp_out = self.mlp(x)
        transformer_out = self.transformer(x)
        
        # Concatenate outputs
        combined = torch.cat([mlp_out, transformer_out], dim=-1)
        
        return self.fusion(combined)


def create_model(
    model_type: str,
    input_dim: int,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function pour créer un modèle.
    
    Args:
        model_type: Type de modèle ('mlp', 'tabnet', 'transformer', 'dann', 'ensemble')
        input_dim: Dimension d'entrée
        num_classes: Nombre de classes
        **kwargs: Arguments supplémentaires pour le modèle
    
    Returns:
        Modèle initialisé
    """
    models = {
        'mlp': MLPClassifier,
        'tabnet': TabNetClassifier,
        'transformer': FTTransformer,
        'dann': DomainAdversarialNetwork,
        'ensemble': EnsembleModel
    }
    
    if model_type not in models:
        raise ValueError(f"Type de modèle inconnu: {model_type}. Choix: {list(models.keys())}")
    
    return models[model_type](input_dim=input_dim, num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Test des modèles
    print("Test des architectures...")
    
    batch_size = 32
    input_dim = 464
    num_classes = 2
    
    x = torch.randn(batch_size, input_dim)
    
    # Test MLP
    print("\n1. MLP Classifier")
    mlp = MLPClassifier(input_dim, num_classes=num_classes)
    out = mlp(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    
    # Test TabNet
    print("\n2. TabNet Classifier")
    tabnet = TabNetClassifier(input_dim, num_classes=num_classes)
    out = tabnet(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in tabnet.parameters()):,}")
    
    # Test Transformer
    print("\n3. FT-Transformer")
    transformer = FTTransformer(input_dim, num_classes=num_classes)
    out = transformer(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # Test DANN
    print("\n4. Domain Adversarial Network")
    dann = DomainAdversarialNetwork(input_dim, num_classes=num_classes)
    label_out, domain_out = dann(x, alpha=0.5)
    print(f"   Input: {x.shape} -> Label: {label_out.shape}, Domain: {domain_out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in dann.parameters()):,}")
    
    print("\n✅ Tous les modèles fonctionnent correctement!")
