#!/usr/bin/env python3
"""
Testing Script for IGH Deep Learning Models

Evaluates a trained model on a test dataset and generates
detailed reports, curves and analyses.

Outputs:
- Full evaluation report (TXT + JSON)
- ROC and Precision-Recall curves
- Confusion matrix
- Probability distribution plots
- False Negative (FN) list
- Decision threshold analysis

Usage:
    python test_deep_bio_classifier.py \
        --model /path/to/final_model.pt \
        --dataset /path/to/test_data.csv \
        --cohort "Patient_Cohort_A" \
        --output /path/to/output_dir

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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score, roc_curve, precision_recall_curve,
    cohen_kappa_score, log_loss
)

from models import (
    MLPClassifier, TabNetClassifier, FTTransformer,
    DomainAdversarialNetwork, create_model
)

warnings.filterwarnings('ignore')

# Configuration du device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelTester:
    """
    Classe pour tester un modèle Deep Learning entraîné.
    
    Génère des rapports détaillés, des courbes et des analyses
    pour évaluer les performances du modèle sur un dataset de test.
    """
    
    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        cohort_name: str,
        output_dir: Optional[str] = None,
        batch_size: int = 256
    ):
        """
        Initialise le testeur de modèle.
        
        Args:
            model_path: Chemin vers le fichier .pt du modèle
            dataset_path: Chemin vers le fichier CSV de test
            cohort_name: Nom de la cohorte de test
            output_dir: Répertoire de sortie pour les résultats
            batch_size: Taille des batches pour l'inférence
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.cohort_name = cohort_name
        self.batch_size = batch_size
        
        # Répertoire de sortie
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"test_results_{cohort_name}_{timestamp}"
        else:
            self.output_dir = output_dir
        
        self._create_output_dirs()
        
        # Composants chargés
        self.model = None
        self.model_type = None
        self.scaler = None
        self.checkpoint = None
        
        # Résultats
        self.predictions = None
        self.probabilities = None
        self.labels = None
        self.sequence_indices = None
        
        print("=" * 60)
        print("🧪 TESTEUR DE MODÈLE DEEP LEARNING IGH")
        print("=" * 60)
        print(f"   Modèle: {model_path}")
        print(f"   Dataset: {dataset_path}")
        print(f"   Cohorte: {cohort_name}")
        print(f"   Device: {DEVICE}")
    
    def _create_output_dirs(self):
        """Crée les répertoires de sortie."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.dirs = {
            'plots': os.path.join(self.output_dir, "plots"),
            'reports': os.path.join(self.output_dir, "reports"),
            'data': os.path.join(self.output_dir, "data")
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def _detect_model_type_from_state_dict(self, state_dict: dict) -> str:
        """
        Détecte automatiquement le type de modèle à partir des clés du state_dict.
        
        Args:
            state_dict: Dictionnaire d'état du modèle
        
        Returns:
            Type de modèle détecté ('mlp', 'tabnet', 'transformer', 'dann')
        """
        keys = list(state_dict.keys())
        keys_str = ' '.join(keys)
        
        # Détecter FTTransformer (contient cls_token, feature_embedding, transformer.layers)
        if 'cls_token' in keys_str or 'feature_embedding' in keys_str or 'transformer.layers' in keys_str:
            return 'transformer'
        
        # Détecter DANN (contient domain_classifier)
        if 'domain_classifier' in keys_str:
            return 'dann'
        
        # Détecter TabNet (contient initial_bn, shared_fc, attention)
        if 'initial_bn' in keys_str or 'shared_fc' in keys_str or 'attention_fcs' in keys_str:
            return 'tabnet'
        
        # Par défaut, c'est un MLP (contient encoder et classifier)
        return 'mlp'
    
    def load_model(self) -> nn.Module:
        """
        Charge le modèle depuis le fichier .pt.
        
        Returns:
            Modèle chargé
        """
        print("\n📦 Chargement du modèle...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
        
        # Charger le checkpoint
        # PyTorch 2.6+ nécessite weights_only=False pour charger les scalers sklearn et objets numpy
        # Ceci est sécurisé car nous faisons confiance aux checkpoints générés par notre propre code
        self.checkpoint = torch.load(self.model_path, map_location=DEVICE, weights_only=False)
        
        # Extraire les informations du checkpoint
        self.model_type = self.checkpoint.get('model_type', None)
        self.scaler = self.checkpoint.get('scaler', None)
        hidden_dims = self.checkpoint.get('hidden_dims', [512, 256, 128, 64])
        dropout = self.checkpoint.get('dropout', 0.3)
        
        # Si le model_type n'est pas dans le checkpoint, le détecter automatiquement
        if self.model_type is None:
            state_dict = self.checkpoint.get('model_state_dict', self.checkpoint)
            self.model_type = self._detect_model_type_from_state_dict(state_dict)
            print(f"   ℹ️ Type de modèle détecté automatiquement: {self.model_type}")
        
        print(f"   - Type de modèle: {self.model_type}")
        print(f"   - Hidden dims: {hidden_dims}")
        print(f"   - Dropout: {dropout}")
        
        if 'best_val_f1' in self.checkpoint:
            print(f"   - Meilleur Val F1 (entraînement): {self.checkpoint['best_val_f1']:.4f}")
        if 'val_f1' in self.checkpoint:
            print(f"   - Val F1 (checkpoint): {self.checkpoint['val_f1']:.4f}")
        
        return self.checkpoint
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Charge et prépare le dataset de test.
        
        Returns:
            X: Features
            y: Labels
            df: DataFrame original (pour récupérer les indices)
        """
        print("\n📂 Chargement du dataset de test...")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset non trouvé: {self.dataset_path}")
        
        df = pd.read_csv(self.dataset_path)
        
        # Supprimer la colonne index si présente
        if df.columns[0] == 'Unnamed: 0' or str(df.columns[0]).isdigit():
            df = df.iloc[:, 1:]
        
        print(f"   - Lignes: {len(df):,}")
        print(f"   - Colonnes: {len(df.columns)}")
        
        # Colonnes à exclure
        cols_to_drop = ['label', 'data_type', 'source', 'sequence_id', 'index']
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        
        # Extraire les features
        feature_cols = [c for c in df.columns if c not in cols_to_drop]
        X = df[feature_cols].values.astype(np.float32)
        
        # Encoder les labels
        if 'label' in df.columns:
            label_map = {'TN': 0, 'TP': 1}
            y = np.array([label_map.get(l, l) for l in df['label'].values]).astype(np.int64)
        else:
            raise ValueError("La colonne 'label' est requise dans le dataset")
        
        # Garder les indices originaux
        if 'sequence_id' in df.columns:
            self.sequence_indices = df['sequence_id'].values
        else:
            self.sequence_indices = np.arange(len(df))
        
        print(f"   - Features: {X.shape[1]}")
        print(f"   - Distribution labels: TP={np.sum(y==1):,}, TN={np.sum(y==0):,}")
        print(f"   - Ratio TP: {np.mean(y)*100:.2f}%")
        
        return X, y, df
    
    def _create_model_from_checkpoint(self, input_dim: int) -> nn.Module:
        """
        Recrée le modèle à partir du checkpoint.
        
        Args:
            input_dim: Dimension d'entrée
        
        Returns:
            Modèle instancié
        """
        hidden_dims = self.checkpoint.get('hidden_dims', [512, 256, 128, 64])
        dropout = self.checkpoint.get('dropout', 0.3)
        
        if self.model_type == 'mlp':
            model = MLPClassifier(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=2,
                dropout=dropout,
                use_batch_norm=True
            )
        elif self.model_type == 'tabnet':
            model = TabNetClassifier(
                input_dim=input_dim,
                num_classes=2,
                n_steps=3,
                n_d=64,
                n_a=64,
                gamma=1.5,
                dropout=dropout
            )
        elif self.model_type == 'transformer':
            model = FTTransformer(
                input_dim=input_dim,
                num_classes=2,
                d_model=128,
                n_heads=4,
                n_layers=3,
                dropout=dropout
            )
        elif self.model_type == 'dann':
            model = DomainAdversarialNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=2,
                dropout=dropout
            )
        else:
            raise ValueError(f"Type de modèle inconnu: {self.model_type}")
        
        return model
    
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Effectue les prédictions sur le dataset.
        
        Args:
            X: Features du dataset de test
        
        Returns:
            predictions: Prédictions (0 ou 1)
            probabilities: Probabilités de la classe positive
        """
        print("\n🔮 Génération des prédictions...")
        
        # Créer le modèle
        self.model = self._create_model_from_checkpoint(input_dim=X.shape[1])
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        # Normaliser les données
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            print("   ⚠️ Scaler non trouvé dans le checkpoint, utilisation des données brutes")
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
        
        # Créer le DataLoader
        dataset = TensorDataset(torch.FloatTensor(X_scaled))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_probs = []
        all_preds = []
        
        for batch in loader:
            X_batch = batch[0].to(DEVICE)
            
            if self.model_type == 'dann':
                outputs, _ = self.model(X_batch, alpha=0)
            else:
                outputs = self.model(X_batch)
            
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        
        self.predictions = np.array(all_preds)
        self.probabilities = np.array(all_probs)
        
        print(f"   - Prédictions générées: {len(self.predictions):,}")
        print(f"   - Distribution prédictions: TP={np.sum(self.predictions==1):,}, TN={np.sum(self.predictions==0):,}")
        
        return self.predictions, self.probabilities
    
    def compute_metrics(self, y_true: np.ndarray) -> Dict:
        """
        Calcule toutes les métriques d'évaluation.
        
        Args:
            y_true: Labels réels
        
        Returns:
            Dictionnaire des métriques
        """
        print("\n📊 Calcul des métriques...")
        
        self.labels = y_true
        
        metrics = {
            # Métriques de base
            'accuracy': accuracy_score(y_true, self.predictions),
            'balanced_accuracy': balanced_accuracy_score(y_true, self.predictions),
            'precision': precision_score(y_true, self.predictions, zero_division=0),
            'recall': recall_score(y_true, self.predictions, zero_division=0),
            'specificity': recall_score(y_true, self.predictions, pos_label=0, zero_division=0),
            'f1_score': f1_score(y_true, self.predictions, zero_division=0),
            
            # Métriques avancées
            'mcc': matthews_corrcoef(y_true, self.predictions),
            'cohen_kappa': cohen_kappa_score(y_true, self.predictions),
            'roc_auc': roc_auc_score(y_true, self.probabilities) if len(np.unique(y_true)) > 1 else 0,
            'pr_auc': average_precision_score(y_true, self.probabilities) if len(np.unique(y_true)) > 1 else 0,
            'log_loss': log_loss(y_true, self.probabilities) if len(np.unique(y_true)) > 1 else 0,
            
            # Counts
            'total_samples': len(y_true),
            'positive_samples': int(np.sum(y_true == 1)),
            'negative_samples': int(np.sum(y_true == 0)),
            'predicted_positive': int(np.sum(self.predictions == 1)),
            'predicted_negative': int(np.sum(self.predictions == 0)),
        }
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, self.predictions)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False Positive Rate
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,  # False Negative Rate
        })
        
        # Afficher les résultats principaux
        print(f"\n   {'Métrique':<25} {'Valeur':>10}")
        print("   " + "-" * 35)
        for key in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'specificity', 
                    'f1_score', 'mcc', 'roc_auc', 'pr_auc']:
            print(f"   {key:<25} {metrics[key]:>10.4f}")
        
        print(f"\n   Matrice de confusion:")
        print(f"   TN={tn:,}  FP={fp:,}")
        print(f"   FN={fn:,}  TP={tp:,}")
        
        return metrics
    
    def find_false_negatives(self) -> pd.DataFrame:
        """
        Identifie les Faux Négatifs (FN).
        
        Returns:
            DataFrame avec les informations sur les FN
        """
        print("\n🔍 Identification des Faux Négatifs...")
        
        # FN = label réel est 1 (TP) mais prédiction est 0 (TN)
        fn_mask = (self.labels == 1) & (self.predictions == 0)
        fn_indices = np.where(fn_mask)[0]
        
        fn_data = {
            'sequence_index': fn_indices,
            'sequence_id': self.sequence_indices[fn_indices] if self.sequence_indices is not None else fn_indices,
            'true_label': self.labels[fn_indices],
            'predicted_label': self.predictions[fn_indices],
            'probability_positive': self.probabilities[fn_indices]
        }
        
        fn_df = pd.DataFrame(fn_data)
        fn_df = fn_df.sort_values('probability_positive', ascending=False)
        
        print(f"   - Nombre de FN: {len(fn_df):,}")
        if len(fn_df) > 0:
            print(f"   - Probabilité moyenne des FN: {fn_df['probability_positive'].mean():.4f}")
            print(f"   - Probabilité max des FN: {fn_df['probability_positive'].max():.4f}")
            print(f"   - Probabilité min des FN: {fn_df['probability_positive'].min():.4f}")
        
        return fn_df
    
    def find_false_positives(self) -> pd.DataFrame:
        """
        Identifie les Faux Positifs (FP).
        
        Returns:
            DataFrame avec les informations sur les FP
        """
        print("\n🔍 Identification des Faux Positifs...")
        
        # FP = label réel est 0 (TN) mais prédiction est 1 (TP)
        fp_mask = (self.labels == 0) & (self.predictions == 1)
        fp_indices = np.where(fp_mask)[0]
        
        fp_data = {
            'sequence_index': fp_indices,
            'sequence_id': self.sequence_indices[fp_indices] if self.sequence_indices is not None else fp_indices,
            'true_label': self.labels[fp_indices],
            'predicted_label': self.predictions[fp_indices],
            'probability_positive': self.probabilities[fp_indices]
        }
        
        fp_df = pd.DataFrame(fp_data)
        fp_df = fp_df.sort_values('probability_positive', ascending=False)
        
        print(f"   - Nombre de FP: {len(fp_df):,}")
        if len(fp_df) > 0:
            print(f"   - Probabilité moyenne des FP: {fp_df['probability_positive'].mean():.4f}")
        
        return fp_df

    def find_true_positives(self) -> pd.DataFrame:
        """
        Identifie les Vrais Positifs (TP).
        
        Returns:
            DataFrame avec les informations sur les TP
        """
        print("\n🔍 Identification des Vrais Positifs...")
        
        # TP = label réel est 1 (TP) et prédiction est 1 (TP)
        tp_mask = (self.labels == 1) & (self.predictions == 1)
        tp_indices = np.where(tp_mask)[0]
        
        tp_data = {
            'sequence_index': tp_indices,
            'sequence_id': self.sequence_indices[tp_indices] if self.sequence_indices is not None else tp_indices,
            'true_label': self.labels[tp_indices],
            'predicted_label': self.predictions[tp_indices],
            'probability_positive': self.probabilities[tp_indices]
        }
        
        tp_df = pd.DataFrame(tp_data)
        tp_df = tp_df.sort_values('probability_positive', ascending=False)
        
        print(f"   - Nombre de TP: {len(tp_df):,}")
        if len(tp_df) > 0:
            print(f"   - Probabilité moyenne des TP: {tp_df['probability_positive'].mean():.4f}")
        
        return tp_df
    
    def plot_roc_curve(self) -> str:
        """
        Génère la courbe ROC.
        
        Returns:
            Chemin vers l'image sauvegardée
        """
        print("\n📈 Génération de la courbe ROC...")
        
        fpr, tpr, thresholds = roc_curve(self.labels, self.probabilities)
        roc_auc = roc_auc_score(self.labels, self.probabilities)
        
        # Trouver le seuil optimal (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        plt.figure(figsize=(10, 8))
        
        # Courbe ROC
        plt.plot(fpr, tpr, color='#3498db', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        
        # Ligne de référence
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
                label='Random classifier')
        
        # Point optimal
        plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], color='red', s=100, 
                   zorder=5, label=f'Optimal threshold = {optimal_threshold:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title(f'ROC Curve - {self.cohort_name}\nModel: {self.model_type.upper()}', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Ajouter les métriques
        textstr = f'Sensitivity: {tpr[optimal_idx]:.3f}\nSpecificity: {1-fpr[optimal_idx]:.3f}'
        plt.annotate(textstr, xy=(0.6, 0.2), fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        save_path = os.path.join(self.dirs['plots'], 'roc_curve.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Sauvegardé: {save_path}")
        
        return save_path
    
    def plot_precision_recall_curve(self) -> str:
        """
        Génère la courbe Precision-Recall.
        
        Returns:
            Chemin vers l'image sauvegardée
        """
        print("\n📈 Génération de la courbe Precision-Recall...")
        
        precision, recall, thresholds = precision_recall_curve(self.labels, self.probabilities)
        pr_auc = average_precision_score(self.labels, self.probabilities)
        
        # Baseline (random classifier)
        baseline = np.sum(self.labels) / len(self.labels)
        
        plt.figure(figsize=(10, 8))
        
        # Courbe PR
        plt.plot(recall, precision, color='#2ecc71', lw=2, 
                label=f'PR curve (AP = {pr_auc:.4f})')
        
        # Baseline
        plt.axhline(y=baseline, color='gray', lw=1, linestyle='--', 
                   label=f'Baseline = {baseline:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {self.cohort_name}\nModel: {self.model_type.upper()}', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.dirs['plots'], 'precision_recall_curve.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Sauvegardé: {save_path}")
        
        return save_path
    
    def plot_confusion_matrix(self) -> str:
        """
        Génère la matrice de confusion.
        
        Returns:
            Chemin vers l'image sauvegardée
        """
        print("\n📈 Génération de la matrice de confusion...")
        
        cm = confusion_matrix(self.labels, self.predictions)
        
        # Version normalisée
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Matrice de confusion (counts)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['TN (Pred)', 'TP (Pred)'], 
                   yticklabels=['TN (True)', 'TP (True)'])
        axes[0].set_title(f'Confusion Matrix (Counts)\n{self.cohort_name}', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=11)
        axes[0].set_xlabel('Predicted Label', fontsize=11)
        
        # Matrice normalisée (pourcentages)
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                   xticklabels=['TN (Pred)', 'TP (Pred)'], 
                   yticklabels=['TN (True)', 'TP (True)'])
        axes[1].set_title(f'Confusion Matrix (Normalized)\n{self.cohort_name}', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=11)
        axes[1].set_xlabel('Predicted Label', fontsize=11)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.dirs['plots'], 'confusion_matrix.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Sauvegardé: {save_path}")
        
        return save_path
    
    def plot_probability_distribution(self) -> str:
        """
        Génère la distribution des probabilités par classe.
        
        Returns:
            Chemin vers l'image sauvegardée
        """
        print("\n📈 Génération de la distribution des probabilités...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogramme par classe
        probs_tn = self.probabilities[self.labels == 0]
        probs_tp = self.probabilities[self.labels == 1]
        
        axes[0].hist(probs_tn, bins=50, alpha=0.7, label=f'TN (n={len(probs_tn):,})', color='#3498db')
        axes[0].hist(probs_tp, bins=50, alpha=0.7, label=f'TP (n={len(probs_tp):,})', color='#e74c3c')
        axes[0].axvline(x=0.5, color='black', linestyle='--', lw=1, label='Threshold=0.5')
        axes[0].set_xlabel('Probability (P(TP))', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title(f'Probability Distribution by Class\n{self.cohort_name}', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # KDE plot
        sns.kdeplot(probs_tn, ax=axes[1], label=f'TN', color='#3498db', fill=True, alpha=0.3)
        sns.kdeplot(probs_tp, ax=axes[1], label=f'TP', color='#e74c3c', fill=True, alpha=0.3)
        axes[1].axvline(x=0.5, color='black', linestyle='--', lw=1, label='Threshold=0.5')
        axes[1].set_xlabel('Probability (P(TP))', fontsize=11)
        axes[1].set_ylabel('Density', fontsize=11)
        axes[1].set_title(f'Probability Density by Class\n{self.cohort_name}', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.dirs['plots'], 'probability_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Sauvegardé: {save_path}")
        
        return save_path
    
    def plot_threshold_analysis(self) -> str:
        """
        Analyse l'impact de différents seuils de décision.
        
        Returns:
            Chemin vers l'image sauvegardée
        """
        print("\n📈 Génération de l'analyse par seuil...")
        
        thresholds = np.linspace(0.01, 0.99, 100)
        
        metrics_by_threshold = {
            'threshold': thresholds,
            'precision': [],
            'recall': [],
            'f1': [],
            'specificity': []
        }
        
        for thresh in thresholds:
            preds_thresh = (self.probabilities >= thresh).astype(int)
            
            metrics_by_threshold['precision'].append(
                precision_score(self.labels, preds_thresh, zero_division=0)
            )
            metrics_by_threshold['recall'].append(
                recall_score(self.labels, preds_thresh, zero_division=0)
            )
            metrics_by_threshold['f1'].append(
                f1_score(self.labels, preds_thresh, zero_division=0)
            )
            metrics_by_threshold['specificity'].append(
                recall_score(self.labels, preds_thresh, pos_label=0, zero_division=0)
            )
        
        # Trouver le seuil optimal pour F1
        f1_array = np.array(metrics_by_threshold['f1'])
        optimal_idx = np.argmax(f1_array)
        optimal_threshold = thresholds[optimal_idx]
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(thresholds, metrics_by_threshold['precision'], 
                label='Precision', color='#3498db', lw=2)
        plt.plot(thresholds, metrics_by_threshold['recall'], 
                label='Recall (Sensitivity)', color='#e74c3c', lw=2)
        plt.plot(thresholds, metrics_by_threshold['f1'], 
                label='F1 Score', color='#2ecc71', lw=2)
        plt.plot(thresholds, metrics_by_threshold['specificity'], 
                label='Specificity', color='#9b59b6', lw=2)
        
        plt.axvline(x=0.5, color='gray', linestyle='--', lw=1, alpha=0.7, 
                   label='Default threshold (0.5)')
        plt.axvline(x=optimal_threshold, color='orange', linestyle='--', lw=2, 
                   label=f'Optimal F1 threshold ({optimal_threshold:.3f})')
        
        plt.xlabel('Decision Threshold', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)
        plt.title(f'Metrics vs Decision Threshold\n{self.cohort_name} - {self.model_type.upper()}', fontsize=14)
        plt.legend(loc='center right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        
        plt.tight_layout()
        
        save_path = os.path.join(self.dirs['plots'], 'threshold_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Sauvegardé: {save_path}")
        print(f"   ℹ️ Seuil optimal pour F1: {optimal_threshold:.3f}")
        
        return save_path
    
    def generate_report(self, metrics: Dict, fn_df: pd.DataFrame, fp_df: pd.DataFrame) -> str:
        """
        Génère un rapport d'évaluation complet.
        
        Args:
            metrics: Dictionnaire des métriques
            fn_df: DataFrame des faux négatifs
            fp_df: DataFrame des faux positifs
        
        Returns:
            Chemin vers le rapport
        """
        print("\n📄 Génération du rapport...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.dirs['reports'], f"evaluation_report_{timestamp}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAPPORT D'ÉVALUATION - MODÈLE DEEP LEARNING IGH\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Cohorte: {self.cohort_name}\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Modèle: {self.model_path}\n")
            f.write(f"Type de modèle: {self.model_type}\n")
            f.write(f"Dataset de test: {self.dataset_path}\n")
            f.write(f"Nombre d'échantillons: {metrics['total_samples']:,}\n")
            f.write(f"  - Positifs (TP): {metrics['positive_samples']:,}\n")
            f.write(f"  - Négatifs (TN): {metrics['negative_samples']:,}\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("MÉTRIQUES DE PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Métrique':<25} {'Valeur':>10}\n")
            f.write("-" * 35 + "\n")
            
            metric_order = [
                ('accuracy', 'Accuracy'),
                ('balanced_accuracy', 'Balanced Accuracy'),
                ('precision', 'Precision'),
                ('recall', 'Recall (Sensitivity)'),
                ('specificity', 'Specificity'),
                ('f1_score', 'F1 Score'),
                ('mcc', 'MCC'),
                ('cohen_kappa', 'Cohen Kappa'),
                ('roc_auc', 'ROC AUC'),
                ('pr_auc', 'PR AUC'),
                ('log_loss', 'Log Loss'),
            ]
            
            for key, name in metric_order:
                f.write(f"{name:<25} {metrics[key]:>10.4f}\n")
            
            f.write("\n")
            f.write("-" * 40 + "\n")
            f.write("MATRICE DE CONFUSION\n")
            f.write("-" * 40 + "\n")
            f.write(f"                  Prédit TN    Prédit TP\n")
            f.write(f"Réel TN           {metrics['true_negatives']:>8,}    {metrics['false_positives']:>8,}\n")
            f.write(f"Réel TP           {metrics['false_negatives']:>8,}    {metrics['true_positives']:>8,}\n\n")
            
            f.write(f"False Positive Rate (FPR): {metrics['fpr']:.4f}\n")
            f.write(f"False Negative Rate (FNR): {metrics['fnr']:.4f}\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("ANALYSE DES ERREURS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Nombre de Faux Négatifs (FN): {len(fn_df):,}\n")
            f.write(f"Nombre de Faux Positifs (FP): {len(fp_df):,}\n")
            
            if len(fn_df) > 0:
                f.write(f"\nProbabilité moyenne des FN: {fn_df['probability_positive'].mean():.4f}\n")
                f.write(f"Probabilité médiane des FN: {fn_df['probability_positive'].median():.4f}\n")
            
            if len(fp_df) > 0:
                f.write(f"\nProbabilité moyenne des FP: {fp_df['probability_positive'].mean():.4f}\n")
                f.write(f"Probabilité médiane des FP: {fp_df['probability_positive'].median():.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("FIN DU RAPPORT\n")
            f.write("=" * 80 + "\n")
        
        # Sauvegarder aussi en JSON
        json_path = os.path.join(self.dirs['reports'], f"metrics_{timestamp}.json")
        
        json_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in metrics.items()}
        json_metrics['cohort_name'] = self.cohort_name
        json_metrics['model_type'] = self.model_type
        json_metrics['model_path'] = self.model_path
        json_metrics['dataset_path'] = self.dataset_path
        json_metrics['timestamp'] = timestamp
        json_metrics['num_false_negatives'] = len(fn_df)
        json_metrics['num_false_positives'] = len(fp_df)
        
        with open(json_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"   ✓ Rapport TXT: {report_path}")
        print(f"   ✓ Rapport JSON: {json_path}")
        
        return report_path
    
    def save_error_analysis(
        self,
        fn_df: pd.DataFrame,
        fp_df: pd.DataFrame,
        tp_df: pd.DataFrame
    ) -> Tuple[str, str, str]:
        """
        Sauvegarde l'analyse des erreurs (FN et FP).
        
        Args:
            fn_df: DataFrame des faux négatifs
            fp_df: DataFrame des faux positifs
            tp_df: DataFrame des vrais positifs
        
        Returns:
            Chemins vers les fichiers sauvegardés
        """
        print("\n💾 Sauvegarde de l'analyse des erreurs...")
        
        # Sauvegarder les FN
        fn_path = os.path.join(self.dirs['data'], 'false_negatives.csv')
        fn_df.to_csv(fn_path, index=False)
        print(f"   ✓ Faux Négatifs: {fn_path}")
        
        # Sauvegarder les FP
        fp_path = os.path.join(self.dirs['data'], 'false_positives.csv')
        fp_df.to_csv(fp_path, index=False)
        print(f"   ✓ Faux Positifs: {fp_path}")

        # Sauvegarder les TP
        tp_path = os.path.join(self.dirs['data'], 'true_positives.csv')
        tp_df.to_csv(tp_path, index=False)
        print(f"   ✓ Vrais Positifs: {tp_path}")
        
        # Fichier TXT simple avec les indices des FN
        fn_txt_path = os.path.join(self.dirs['data'], 'false_negatives_indices.txt')
        with open(fn_txt_path, 'w') as f:
            f.write(f"# Liste des séquences Faux Négatifs (FN)\n")
            f.write(f"# Cohorte: {self.cohort_name}\n")
            f.write(f"# Total FN: {len(fn_df)}\n")
            f.write(f"# Format: sequence_id,probability\n")
            f.write("#" + "-" * 40 + "\n")
            for _, row in fn_df.iterrows():
                f.write(f"{row['sequence_id']},{row['probability_positive']:.4f}\n")
        
        print(f"   ✓ Indices FN (TXT): {fn_txt_path}")
        
        # Sauvegarder toutes les prédictions
        all_preds_df = pd.DataFrame({
            'sequence_index': np.arange(len(self.labels)),
            'sequence_id': self.sequence_indices,
            'true_label': self.labels,
            'predicted_label': self.predictions,
            'probability_positive': self.probabilities,
            'correct': self.labels == self.predictions
        })
        
        all_preds_path = os.path.join(self.dirs['data'], 'all_predictions.csv')
        all_preds_df.to_csv(all_preds_path, index=False)
        print(f"   ✓ Toutes les prédictions: {all_preds_path}")
        
        return fn_path, fp_path, tp_path
    
    def run(self) -> Dict:
        """
        Exécute le pipeline complet de test.
        
        Returns:
            Dictionnaire des métriques finales
        """
        print("\n" + "=" * 60)
        print("🚀 DÉMARRAGE DU TEST")
        print("=" * 60)
        
        # 1. Charger le modèle
        self.load_model()
        
        # 2. Charger le dataset
        X, y, df = self.load_dataset()
        
        # 3. Générer les prédictions
        self.predict(X)
        
        # 4. Calculer les métriques
        metrics = self.compute_metrics(y)
        
        # 5. Identifier les erreurs
        fn_df = self.find_false_negatives()
        fp_df = self.find_false_positives()
        tp_df = self.find_true_positives()
        
        # 6. Générer les graphiques
        print("\n" + "-" * 40)
        print("GÉNÉRATION DES VISUALISATIONS")
        print("-" * 40)
        
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_confusion_matrix()
        self.plot_probability_distribution()
        self.plot_threshold_analysis()
        
        # 7. Générer le rapport
        self.generate_report(metrics, fn_df, fp_df)
        
        # 8. Sauvegarder l'analyse des erreurs
        self.save_error_analysis(fn_df, fp_df, tp_df)
        
        # Résumé final
        print("\n" + "=" * 60)
        print("✅ TEST TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        print(f"\n📁 Résultats sauvegardés dans: {self.output_dir}")
        print(f"\n📊 Résumé des performances:")
        print(f"   - Accuracy: {metrics['accuracy']:.4f}")
        print(f"   - F1 Score: {metrics['f1_score']:.4f}")
        print(f"   - ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"   - MCC: {metrics['mcc']:.4f}")
        print(f"\n⚠️ Erreurs:")
        print(f"   - Faux Négatifs: {len(fn_df):,}")
        print(f"   - Faux Positifs: {len(fp_df):,}")
        
        return metrics


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description='Test d\'un modèle Deep Learning IGH',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
    python test_deep_bio_classifier.py \\
        --model /raid/datasets/igh/train_data/deep_pipeline/results_transformer_real50/models/final_model.pt \\
        --dataset /raid/datasets/igh/train_data/df_patient_test.csv \\
        --cohort "Patient_Test_A"

    python test_deep_bio_classifier.py -m model.pt -d test.csv -c "Cohorte_1" -o results/
        """
    )
    
    parser.add_argument('-m', '--model', type=str, required=True,
                       help='Chemin vers le fichier .pt du modèle entraîné')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                       help='Chemin vers le fichier CSV du dataset de test')
    parser.add_argument('-c', '--cohort', type=str, required=True,
                       help='Nom de la cohorte de test')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Répertoire de sortie pour les résultats (optionnel)')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                       help='Taille des batches pour l\'inférence (défaut: 256)')
    
    args = parser.parse_args()
    
    # Créer et exécuter le testeur
    tester = ModelTester(
        model_path=args.model,
        dataset_path=args.dataset,
        cohort_name=args.cohort,
        output_dir=args.output,
        batch_size=args.batch_size
    )
    
    metrics = tester.run()
    
    return metrics


if __name__ == "__main__":
    main()
