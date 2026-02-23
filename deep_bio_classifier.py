#!/usr/bin/env python3
"""
Deep Learning Pipeline for IGH Biological Sequence Classification

Implements a full training pipeline with:
- Support for multiple architectures (MLP, TabNet, Transformer, DANN)
- Mixed real and synthetic training data
- Early stopping and checkpointing
- Detailed metrics and visualizations

Usage:
    python deep_bio_classifier.py \
        --data /path/to/combined_data.csv \
        --output /path/to/output_dir \
        --model transformer \
        --real-ratio 0.5 \
        --epochs 100

Author: Hadrien Gayap
Date: 2025
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score
)

from models import (
    MLPClassifier, TabNetClassifier, FTTransformer,
    DomainAdversarialNetwork, create_model
)
from losses import FocalLoss, LabelSmoothingLoss, DomainAdversarialLoss, get_loss_function

warnings.filterwarnings('ignore')

# Configuration du device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")


def _compute_stratified_counts(
    counts: pd.Series,
    n_samples: int,
    rng: np.random.RandomState
) -> Dict[str, int]:
    if n_samples > counts.sum():
        raise ValueError("n_samples dépasse la taille disponible pour la stratification")

    proportions = counts / counts.sum()
    raw = proportions * n_samples
    base = np.floor(raw).astype(int)
    remainder = n_samples - base.sum()

    fractional = (raw - base).sort_values(ascending=False)
    desired = base.copy()
    capacity = counts - desired

    labels_order = list(fractional.index)
    idx = 0
    while remainder > 0:
        label = labels_order[idx % len(labels_order)]
        if capacity[label] > 0:
            desired[label] += 1
            capacity[label] -= 1
            remainder -= 1
        idx += 1
        if idx > len(labels_order) * 2 and remainder > 0:
            available = capacity[capacity > 0].index.tolist()
            if not available:
                break
            label = rng.choice(available)
            desired[label] += 1
            capacity[label] -= 1
            remainder -= 1

    if remainder > 0:
        raise ValueError("Impossible de répartir les échantillons sans dépasser les capacités")

    return desired.to_dict()


def _stratified_sample(
    df: pd.DataFrame,
    n_samples: int,
    label_col: str,
    random_state: int
) -> pd.DataFrame:
    if n_samples == 0:
        return df.iloc[0:0].copy()
    if n_samples > len(df):
        raise ValueError("n_samples dépasse la taille du DataFrame")

    rng = np.random.RandomState(random_state)
    counts = df[label_col].value_counts()
    desired = _compute_stratified_counts(counts, n_samples, rng)

    parts = []
    for label, count in desired.items():
        if count == 0:
            continue
        group = df[df[label_col] == label]
        parts.append(group.sample(n=count, random_state=rng.randint(0, 2**32 - 1)))

    if not parts:
        return df.iloc[0:0].copy()

    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(frac=1, random_state=rng.randint(0, 2**32 - 1)).reset_index(drop=True)


def build_fixed_cohort_dataset(
    df: pd.DataFrame,
    n_real_total: int,
    synthetic_pct: float,
    random_state: int = 42,
    always_include_source: str = "cll_dna",
    always_include_n: int = 20000,
    stratify: bool = True
) -> pd.DataFrame:
    _ensure_columns(df, ["data_type", "source", "label"])

    df_real_all = df[df["data_type"] == "real"].copy()
    df_synth_all = df[df["data_type"] == "synthetic"].copy()

    if len(df_real_all) < n_real_total:
        raise ValueError("Pas assez de données réelles pour la cohorte demandée")

    df_real_cll = df_real_all[df_real_all["source"] == always_include_source]
    if len(df_real_cll) < always_include_n:
        raise ValueError(
            f"Nombre insuffisant de {always_include_source}: {len(df_real_cll)} (attendu {always_include_n})"
        )
    if len(df_real_cll) > always_include_n:
        df_real_cll = df_real_cll.sample(n=always_include_n, random_state=random_state)

    n_real_remaining = n_real_total - len(df_real_cll)
    if n_real_remaining < 0:
        raise ValueError("n_real_total est inférieur au nombre imposé de cll_dna")

    df_real_other = df_real_all[df_real_all["source"] != always_include_source]
    if n_real_remaining > len(df_real_other):
        raise ValueError("Pas assez de données réelles (hors cll_dna) pour compléter la cohorte")

    if stratify:
        df_real_other_sampled = _stratified_sample(
            df_real_other, n_real_remaining, label_col="label", random_state=random_state + 1
        )
    else:
        df_real_other_sampled = df_real_other.sample(n=n_real_remaining, random_state=random_state + 1)

    df_real_final = pd.concat([df_real_cll, df_real_other_sampled], ignore_index=True)

    n_synth = int(round(n_real_total * (synthetic_pct / 100.0)))
    if n_synth > len(df_synth_all):
        raise ValueError("Pas assez de données synthétiques pour la cohorte demandée")

    if stratify:
        df_synth_sampled = _stratified_sample(
            df_synth_all, n_synth, label_col="label", random_state=random_state + 2
        )
    else:
        df_synth_sampled = df_synth_all.sample(n=n_synth, random_state=random_state + 2)

    df_cohort = pd.concat([df_real_final, df_synth_sampled], ignore_index=True)
    df_cohort = df_cohort.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_cohort


class EarlyStopping:
    """Early Stopping pour éviter le surapprentissage."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class DeepBioClassifier:
    """
    Pipeline complet pour la classification biologique avec Deep Learning.
    """
    
    def __init__(
        self,
        model_type: str = 'transformer',
        hidden_dims: List[int] = [512, 256, 128, 64],
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 100,
        patience: int = 15,
        loss_type: str = 'focal',
        output_dir: str = 'results'
    ):
        """
        Initialise le classificateur.
        
        Args:
            model_type: Type de modèle ('mlp', 'tabnet', 'transformer', 'dann')
            hidden_dims: Dimensions des couches cachées (pour MLP/DANN)
            dropout: Taux de dropout
            learning_rate: Taux d'apprentissage
            weight_decay: Régularisation L2
            batch_size: Taille des batches
            epochs: Nombre maximum d'époques
            patience: Patience pour early stopping
            loss_type: Type de fonction de perte
            output_dir: Répertoire de sortie
        """
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.loss_type = loss_type
        self.output_dir = output_dir
        
        # Créer les répertoires de sortie
        self._create_output_dirs()
        
        # Composants
        self.model = None
        self.scaler = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Historique d'entraînement
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'train_auc': [], 'val_auc': [],
            'lr': []
        }
        
        # Meilleur modèle
        self.best_val_f1 = 0
        self.best_model_state = None
        
        print("=" * 60)
        print("🧬 DEEP BIO CLASSIFIER - IGH")
        print("=" * 60)
        print(f"   Modèle: {model_type}")
        print(f"   Device: {DEVICE}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
    
    def _create_output_dirs(self):
        """Crée les répertoires de sortie."""
        self.dirs = {
            'checkpoints': os.path.join(self.output_dir, "checkpoints"),
            'models': os.path.join(self.output_dir, "models"),
            'plots': os.path.join(self.output_dir, "plots"),
            'reports': os.path.join(self.output_dir, "reports")
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def load_and_prepare_data(
        self,
        data_path: Optional[str] = None,
        data_df: Optional[pd.DataFrame] = None,
        real_ratio: float = 0.5,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Charge et prépare les données.
        
        Args:
            data_path: Chemin vers le fichier CSV
            real_ratio: Ratio de données réelles dans l'entraînement
            val_size: Taille du set de validation
            test_size: Taille du set de test
            random_state: Graine aléatoire
        
        Returns:
            train_loader, val_loader, test_loader
        """
        if data_df is not None:
            df = data_df.copy()
            print("\n📂 Chargement des données: DataFrame fourni")
        else:
            if data_path is None:
                raise ValueError("data_path est requis si data_df est absent")
            print(f"\n📂 Chargement des données: {data_path}")
            df = pd.read_csv(data_path)
        
        # Supprimer la colonne index si présente
        if df.columns[0] == 'Unnamed: 0' or str(df.columns[0]).isdigit():
            df = df.iloc[:, 1:]
        
        print(f"   - Total samples: {len(df):,}")
        
        # Séparer les données réelles et synthétiques
        if 'data_type' in df.columns:
            df_real = df[df['data_type'] == 'real'].copy()
            df_synth = df[df['data_type'] == 'synthetic'].copy()
            
            print(f"   - Données réelles: {len(df_real):,}")
            print(f"   - Données synthétiques: {len(df_synth):,}")
            
            # Calculer le nombre de samples réels à utiliser
            n_real = int(len(df_synth) * real_ratio / (1 - real_ratio)) if real_ratio < 1 else len(df_real)
            n_real = min(n_real, len(df_real))
            
            # Échantillonner les données réelles
            if n_real < len(df_real):
                df_real_sampled = df_real.sample(n=n_real, random_state=random_state)
            else:
                df_real_sampled = df_real
            
            # Combiner
            df_combined = pd.concat([df_synth, df_real_sampled], ignore_index=True)
            df_combined = df_combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            print(f"   - Dataset final: {len(df_combined):,} (real_ratio={n_real/len(df_combined)*100:.1f}%)")
        else:
            df_combined = df
            print(f"   - Pas de colonne 'data_type', utilisation de toutes les données")
        
        # Colonnes à exclure
        cols_to_drop = ['label', 'data_type', 'source', 'sequence_id', 'index']
        cols_to_drop = [c for c in cols_to_drop if c in df_combined.columns]
        
        # Features et labels
        feature_cols = [c for c in df_combined.columns if c not in cols_to_drop]
        X = df_combined[feature_cols].values.astype(np.float32)
        
        # Encoder les labels
        label_map = {'TN': 0, 'TP': 1}
        y = np.array([label_map.get(l, l) for l in df_combined['label'].values]).astype(np.int64)
        
        # Domain labels (si DANN)
        if self.model_type == 'dann' and 'data_type' in df_combined.columns:
            domain_map = {'synthetic': 0, 'real': 1}
            self.domain_labels = np.array([domain_map.get(d, 0) for d in df_combined['data_type'].values]).astype(np.float32)
        else:
            self.domain_labels = None
        
        print(f"   - Features: {X.shape[1]}")
        print(f"   - Distribution labels: TP={np.sum(y==1):,}, TN={np.sum(y==0):,}")
        
        # Split train/val/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_size + test_size), random_state=random_state, stratify=y
        )
        
        relative_test_size = test_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=relative_test_size, random_state=random_state, stratify=y_temp
        )
        
        print(f"   - Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
        
        # Normalisation
        self.scaler = RobustScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # Convertir en tenseurs
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.LongTensor(y_test)
        
        # Créer les datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        
        # Weighted sampler pour gérer le déséquilibre
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            sampler=sampler, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        # Stocker les dimensions
        self.input_dim = X_train.shape[1]
        self.num_classes = len(np.unique(y))
        
        # Calculer les poids de classe pour la loss
        self.class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        
        return train_loader, val_loader, test_loader
    
    def build_model(self):
        """Construit le modèle, l'optimiseur et le scheduler."""
        print(f"\n🏗️ Construction du modèle: {self.model_type}")
        
        # Créer le modèle
        if self.model_type == 'mlp':
            self.model = MLPClassifier(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                num_classes=self.num_classes,
                dropout=self.dropout,
                use_batch_norm=True
            )
        elif self.model_type == 'tabnet':
            self.model = TabNetClassifier(
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                n_steps=3,
                n_d=64,
                n_a=64,
                gamma=1.5,
                dropout=self.dropout
            )
        elif self.model_type == 'transformer':
            self.model = FTTransformer(
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                d_model=128,
                n_heads=4,
                n_layers=3,
                d_ff=256,
                dropout=self.dropout
            )
        elif self.model_type == 'dann':
            self.model = DomainAdversarialNetwork(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                num_classes=self.num_classes,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Type de modèle inconnu: {self.model_type}")
        
        self.model = self.model.to(DEVICE)
        
        # Compter les paramètres
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"   - Paramètres: {num_params:,}")
        
        # Loss function
        if self.model_type == 'dann':
            self.criterion = DomainAdversarialLoss(
                class_criterion=FocalLoss(alpha=0.25, gamma=2.0),
                domain_weight=0.1
            )
        else:
            self.criterion = get_loss_function(
                self.loss_type,
                num_classes=self.num_classes
            )
        
        # Optimiseur
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Entraîne le modèle pour une époque."""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Alpha pour DANN (augmente progressivement)
        p = epoch / self.epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            self.optimizer.zero_grad()
            
            if self.model_type == 'dann':
                # Générer des domain labels aléatoires pour le batch
                domain_labels = torch.zeros(X.shape[0], device=DEVICE)
                domain_labels[:X.shape[0]//2] = 1  # Simuler moitié synth, moitié real
                
                label_out, domain_out = self.model(X, alpha=alpha)
                loss, class_loss, domain_loss = self.criterion(
                    label_out, y, domain_out, domain_labels
                )
                outputs = label_out
            else:
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        # Calculer les métriques
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Évalue le modèle sur un dataset."""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            if self.model_type == 'dann':
                outputs, _ = self.model(X, alpha=0)
            else:
                outputs = self.model(X)
            
            if self.model_type == 'dann':
                loss = nn.CrossEntropyLoss()(outputs, y)
            else:
                loss = self.criterion(outputs, y)
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics = {
            'loss': total_loss / len(loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0,
            'mcc': matthews_corrcoef(all_labels, all_preds),
            'balanced_acc': balanced_accuracy_score(all_labels, all_preds)
        }
        
        # Matrice de confusion
        cm = confusion_matrix(all_labels, all_preds)
        metrics['confusion_matrix'] = cm
        metrics['predictions'] = all_preds
        metrics['labels'] = all_labels
        metrics['probabilities'] = all_probs
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """
        Entraîne le modèle.
        
        Args:
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
        """
        print("\n🚀 Démarrage de l'entraînement...")
        print("-" * 60)
        
        early_stopping = EarlyStopping(patience=self.patience, mode='max')
        
        for epoch in range(self.epochs):
            # Entraînement
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Mise à jour du scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Enregistrer l'historique
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['lr'].append(current_lr)
            
            # Afficher les métriques
            print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                  f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                  f"F1: {train_metrics['f1']:.4f}/{val_metrics['f1']:.4f} | "
                  f"AUC: {train_metrics['auc']:.4f}/{val_metrics['auc']:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            # Sauvegarder le meilleur modèle
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_model_state = self.model.state_dict().copy()
                
                # Sauvegarder le checkpoint
                self._save_checkpoint(epoch, val_metrics)
                print(f"         ✓ Nouveau meilleur modèle (F1={val_metrics['f1']:.4f})")
            
            # Early stopping
            if early_stopping(val_metrics['f1']):
                print(f"\n⏹️ Early stopping à l'époque {epoch+1}")
                break
        
        # Charger le meilleur modèle
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print("-" * 60)
        print(f"✅ Entraînement terminé. Meilleur F1: {self.best_val_f1:.4f}")
    
    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Sauvegarde un checkpoint du modèle."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_f1': metrics['f1'],
            'val_auc': metrics['auc'],
            'scaler': self.scaler,
            'model_type': self.model_type,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes
        }
        
        path = os.path.join(self.dirs['checkpoints'], 'best_model.pt')
        torch.save(checkpoint, path)
    
    def save_final_model(self):
        """Sauvegarde le modèle final."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'model_type': self.model_type,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'best_val_f1': self.best_val_f1
        }
        
        path = os.path.join(self.dirs['models'], 'final_model.pt')
        torch.save(checkpoint, path)
        print(f"💾 Modèle final sauvegardé: {path}")
    
    def plot_training_history(self):
        """Génère les courbes d'entraînement."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1 Score
        axes[0, 1].plot(epochs, self.history['train_f1'], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.history['val_f1'], 'r-', label='Validation')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 0].plot(epochs, self.history['train_auc'], 'b-', label='Train')
        axes[1, 0].plot(epochs, self.history['val_auc'], 'r-', label='Validation')
        axes[1, 0].set_title('ROC AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(epochs, self.history['lr'], 'g-')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training History - {self.model_type.upper()}', fontsize=14)
        plt.tight_layout()
        
        path = os.path.join(self.dirs['plots'], 'training_history.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Courbes sauvegardées: {path}")
    
    def plot_confusion_matrix(self, metrics: Dict, title: str = 'Test'):
        """Génère la matrice de confusion."""
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['TN', 'TP'], yticklabels=['TN', 'TP'])
        plt.title(f'Confusion Matrix - {title}\n{self.model_type.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        path = os.path.join(self.dirs['plots'], 'confusion_matrix.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Matrice de confusion sauvegardée: {path}")
    
    def generate_report(self, test_metrics: Dict):
        """Génère un rapport d'évaluation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Rapport textuel
        report_path = os.path.join(self.dirs['reports'], f'deep_report_{timestamp}.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("RAPPORT D'ENTRAÎNEMENT - DEEP LEARNING IGH\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Modèle: {self.model_type}\n")
            f.write(f"Epochs: {len(self.history['train_loss'])}\n\n")
            
            f.write("MÉTRIQUES SUR LE TEST SET\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy:        {test_metrics['accuracy']:.4f}\n")
            f.write(f"Balanced Acc:    {test_metrics['balanced_acc']:.4f}\n")
            f.write(f"Precision:       {test_metrics['precision']:.4f}\n")
            f.write(f"Recall:          {test_metrics['recall']:.4f}\n")
            f.write(f"F1 Score:        {test_metrics['f1']:.4f}\n")
            f.write(f"ROC AUC:         {test_metrics['auc']:.4f}\n")
            f.write(f"MCC:             {test_metrics['mcc']:.4f}\n\n")
            
            f.write("MATRICE DE CONFUSION\n")
            f.write("-" * 40 + "\n")
            cm = test_metrics['confusion_matrix']
            f.write(f"TN={cm[0,0]:5d}  FP={cm[0,1]:5d}\n")
            f.write(f"FN={cm[1,0]:5d}  TP={cm[1,1]:5d}\n")
        
        print(f"📄 Rapport sauvegardé: {report_path}")
        
        # Rapport JSON
        json_path = os.path.join(self.dirs['reports'], f'metrics_{timestamp}.json')
        
        json_metrics = {
            'model_type': self.model_type,
            'timestamp': timestamp,
            'epochs_trained': len(self.history['train_loss']),
            'best_val_f1': float(self.best_val_f1),
            'test_accuracy': float(test_metrics['accuracy']),
            'test_precision': float(test_metrics['precision']),
            'test_recall': float(test_metrics['recall']),
            'test_f1': float(test_metrics['f1']),
            'test_auc': float(test_metrics['auc']),
            'test_mcc': float(test_metrics['mcc']),
            'test_balanced_acc': float(test_metrics['balanced_acc'])
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"📄 Métriques JSON sauvegardées: {json_path}")
    
    def run(
        self,
        data_path: Optional[str] = None,
        real_ratio: float = 0.5,
        data_df: Optional[pd.DataFrame] = None,
        random_state: int = 42
    ):
        """
        Exécute le pipeline complet.
        
        Args:
            data_path: Chemin vers les données
            real_ratio: Ratio de données réelles
            data_df: DataFrame optionnel des données
            random_state: Graine aléatoire pour le split
        """
        # 1. Charger les données
        train_loader, val_loader, test_loader = self.load_and_prepare_data(
            data_path=data_path, data_df=data_df, real_ratio=real_ratio, random_state=random_state
        )
        
        # 2. Construire le modèle
        self.build_model()
        
        # 3. Entraîner
        self.train(train_loader, val_loader)
        
        # 4. Évaluer sur le test set
        print("\n📊 Évaluation sur le test set...")
        test_metrics = self.evaluate(test_loader)
        
        print(f"\n{'Métrique':<20} {'Valeur':>10}")
        print("-" * 30)
        print(f"{'Accuracy':<20} {test_metrics['accuracy']:>10.4f}")
        print(f"{'Balanced Accuracy':<20} {test_metrics['balanced_acc']:>10.4f}")
        print(f"{'Precision':<20} {test_metrics['precision']:>10.4f}")
        print(f"{'Recall':<20} {test_metrics['recall']:>10.4f}")
        print(f"{'F1 Score':<20} {test_metrics['f1']:>10.4f}")
        print(f"{'ROC AUC':<20} {test_metrics['auc']:>10.4f}")
        print(f"{'MCC':<20} {test_metrics['mcc']:>10.4f}")
        
        # 5. Générer les visualisations
        self.plot_training_history()
        self.plot_confusion_matrix(test_metrics)
        
        # 6. Sauvegarder le modèle et le rapport
        self.save_final_model()
        self.generate_report(test_metrics)
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        
        return test_metrics


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description='Deep Learning Pipeline pour Classification IGH',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Arguments obligatoires
    parser.add_argument('-d', '--data', type=str, required=True,
                       help='Chemin vers le fichier CSV des données')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Répertoire de sortie')
    
    # Arguments du modèle
    parser.add_argument('-m', '--model', type=str, default='transformer',
                       choices=['mlp', 'tabnet', 'transformer', 'dann'],
                       help='Type de modèle (default: transformer)')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 256, 128, 64],
                       help='Dimensions des couches cachées')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Taux de dropout (default: 0.3)')
    
    # Arguments d'entraînement
    parser.add_argument('-r', '--real-ratio', type=float, default=0.5,
                       help='Ratio de données réelles (default: 0.5)')
    parser.add_argument('-e', '--epochs', type=int, default=150,
                       help='Nombre d\'époques (default: 150)')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                       help='Taille des batches (default: 256)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                       help='Taux d\'apprentissage (default: 1e-3)')
    parser.add_argument('--patience', type=int, default=50,
                       help='Patience pour early stopping (default: 50)')
    parser.add_argument('--loss', type=str, default='focal',
                       choices=['ce', 'focal', 'label_smoothing', 'combined'],
                       help='Type de loss (default: focal)')
    
    args = parser.parse_args()
    
    # Créer et exécuter le classificateur
    classifier = DeepBioClassifier(
        model_type=args.model,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        loss_type=args.loss,
        output_dir=args.output
    )
    
    classifier.run(args.data, real_ratio=args.real_ratio)


if __name__ == "__main__":
    main()
