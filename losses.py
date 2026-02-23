"""
Fonctions de perte personnalisées pour classification biologique.

Losses disponibles:
- FocalLoss: Pour gérer le déséquilibre des classes
- LabelSmoothingLoss: Pour améliorer la généralisation
- DomainAdversarialLoss: Pour l'adaptation de domaine
- CombinedLoss: Combinaison de plusieurs losses

Auteur: Hadrien Gayap & Assistant IA
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss pour gérer le déséquilibre des classes.
    
    La Focal Loss réduit le poids des exemples bien classifiés
    et se concentre sur les exemples difficiles.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reference: https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Poids pour la classe positive (default: 0.25)
        gamma: Facteur de focalisation (default: 2.0)
        reduction: Type de réduction ('mean', 'sum', 'none')
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la Focal Loss.
        
        Args:
            inputs: Logits du modèle (batch_size, num_classes)
            targets: Labels (batch_size,)
        
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Calcul de alpha_t
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=inputs.device),
            torch.tensor(1 - self.alpha, device=inputs.device)
        )
        
        # Focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss pour améliorer la généralisation.
    
    Au lieu d'utiliser des labels one-hot stricts (0 ou 1),
    utilise des labels adoucis (epsilon et 1-epsilon).
    
    Args:
        num_classes: Nombre de classes
        smoothing: Facteur de lissage (default: 0.1)
        reduction: Type de réduction ('mean', 'sum', 'none')
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la Label Smoothing Loss.
        
        Args:
            inputs: Logits du modèle (batch_size, num_classes)
            targets: Labels (batch_size,)
        
        Returns:
            Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Créer les labels lissés
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # KL divergence
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DomainAdversarialLoss(nn.Module):
    """
    Loss pour Domain Adversarial Training.
    
    Combine la loss de classification avec la loss de domaine
    pour apprendre des features invariantes au domaine.
    
    Args:
        class_criterion: Critère pour la classification
        domain_weight: Poids de la loss de domaine
    """
    
    def __init__(
        self,
        class_criterion: Optional[nn.Module] = None,
        domain_weight: float = 0.1
    ):
        super().__init__()
        self.class_criterion = class_criterion or nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCEWithLogitsLoss()
        self.domain_weight = domain_weight
    
    def forward(
        self,
        class_output: torch.Tensor,
        class_target: torch.Tensor,
        domain_output: torch.Tensor,
        domain_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcule la loss combinée.
        
        Args:
            class_output: Prédictions de classe
            class_target: Labels de classe
            domain_output: Prédictions de domaine
            domain_target: Labels de domaine
        
        Returns:
            total_loss, class_loss, domain_loss
        """
        class_loss = self.class_criterion(class_output, class_target)
        domain_loss = self.domain_criterion(domain_output, domain_target)
        
        total_loss = class_loss + self.domain_weight * domain_loss
        
        return total_loss, class_loss, domain_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss pour classification multi-label.
    
    Permet de traiter différemment les faux positifs et faux négatifs.
    
    Reference: https://arxiv.org/abs/2009.14119
    
    Args:
        gamma_neg: Gamma pour les exemples négatifs
        gamma_pos: Gamma pour les exemples positifs
        clip: Valeur de clip pour les probabilités
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule l'Asymmetric Loss.
        
        Args:
            inputs: Logits du modèle
            targets: Labels one-hot ou probabilités
        
        Returns:
            Asymmetric loss
        """
        # Convertir en probabilités
        probs = torch.sigmoid(inputs)
        
        # One-hot encoding si nécessaire
        if targets.dim() == 1:
            targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1]).float()
        else:
            targets_one_hot = targets
        
        # Calcul des probabilités positives et négatives
        probs_pos = probs
        probs_neg = 1 - probs
        
        # Clipping asymétrique
        probs_neg = (probs_neg + self.clip).clamp(max=1)
        
        # Loss positive et négative
        loss_pos = targets_one_hot * torch.log(probs_pos.clamp(min=1e-8))
        loss_neg = (1 - targets_one_hot) * torch.log(probs_neg.clamp(min=1e-8))
        
        # Focal weighting
        loss_pos = loss_pos * ((1 - probs_pos) ** self.gamma_pos)
        loss_neg = loss_neg * (probs_pos ** self.gamma_neg)
        
        loss = -(loss_pos + loss_neg)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combinaison de plusieurs fonctions de perte.
    
    Args:
        losses: Liste de tuples (loss_fn, weight)
    """
    
    def __init__(self, losses: list):
        super().__init__()
        self.losses = nn.ModuleList([loss for loss, _ in losses])
        self.weights = [weight for _, weight in losses]
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la loss combinée.
        
        Args:
            inputs: Logits du modèle
            targets: Labels
        
        Returns:
            Combined loss
        """
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(inputs, targets)
        
        return total_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss pour l'apprentissage de représentations.
    
    Rapproche les représentations de la même classe
    et éloigne celles de classes différentes.
    
    Args:
        temperature: Température pour le scaling
        base_temperature: Température de base
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la Supervised Contrastive Loss.
        
        Args:
            features: Features normalisées (batch_size, feature_dim)
            labels: Labels (batch_size,)
        
        Returns:
            Contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normaliser les features
        features = F.normalize(features, p=2, dim=1)
        
        # Masque des labels identiques
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Similarité cosinus
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Masque pour exclure la diagonale
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        # Log-softmax pour la stabilité numérique
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Moyenne sur les positifs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        return loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss pour segmentation/classification.
    
    Mesure le chevauchement entre les prédictions et les labels.
    
    Args:
        smooth: Facteur de lissage pour éviter la division par zéro
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la Dice Loss.
        
        Args:
            inputs: Logits du modèle
            targets: Labels
        
        Returns:
            Dice loss
        """
        probs = F.softmax(inputs, dim=1)
        
        # One-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        targets_one_hot = targets_one_hot.permute(0, 1).contiguous()
        
        # Intersection et union
        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        # Dice coefficient
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


def get_loss_function(
    loss_type: str = 'focal',
    num_classes: int = 2,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function pour obtenir une fonction de perte.
    
    Args:
        loss_type: Type de loss ('ce', 'focal', 'label_smoothing', 'asymmetric', 'combined')
        num_classes: Nombre de classes
        class_weights: Poids des classes optionnels
        **kwargs: Arguments supplémentaires
    
    Returns:
        Fonction de perte
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('alpha', 0.25),
            gamma=kwargs.get('gamma', 2.0)
        )
    
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(
            num_classes=num_classes,
            smoothing=kwargs.get('smoothing', 0.1)
        )
    
    elif loss_type == 'asymmetric':
        return AsymmetricLoss(
            gamma_neg=kwargs.get('gamma_neg', 4.0),
            gamma_pos=kwargs.get('gamma_pos', 1.0)
        )
    
    elif loss_type == 'combined':
        losses = [
            (FocalLoss(alpha=0.25, gamma=2.0), 0.7),
            (LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1), 0.3)
        ]
        return CombinedLoss(losses)
    
    elif loss_type == 'dice':
        return DiceLoss(smooth=kwargs.get('smooth', 1.0))
    
    else:
        raise ValueError(f"Type de loss inconnu: {loss_type}")


if __name__ == "__main__":
    # Test des fonctions de perte
    print("Test des fonctions de perte...")
    
    batch_size = 32
    num_classes = 2
    
    # Données de test
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test Focal Loss
    print("\n1. Focal Loss")
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal(inputs, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test Label Smoothing
    print("\n2. Label Smoothing Loss")
    label_smooth = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
    loss = label_smooth(inputs, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test Asymmetric Loss
    print("\n3. Asymmetric Loss")
    asym = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
    loss = asym(inputs, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test Combined Loss
    print("\n4. Combined Loss")
    combined = get_loss_function('combined', num_classes=num_classes)
    loss = combined(inputs, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test Domain Adversarial Loss
    print("\n5. Domain Adversarial Loss")
    domain_output = torch.randn(batch_size)
    domain_target = torch.randint(0, 2, (batch_size,)).float()
    
    dann_loss = DomainAdversarialLoss(domain_weight=0.1)
    total, class_l, domain_l = dann_loss(inputs, targets, domain_output, domain_target)
    print(f"   Total: {total.item():.4f}, Class: {class_l.item():.4f}, Domain: {domain_l.item():.4f}")
    
    # Test Contrastive Loss
    print("\n6. Contrastive Loss")
    features = torch.randn(batch_size, 128)
    contrastive = ContrastiveLoss(temperature=0.07)
    loss = contrastive(features, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n✅ Tous les tests passés!")
