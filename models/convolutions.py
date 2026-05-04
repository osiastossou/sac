"""
convolutions.py
===============
Implémentation des 9 opérateurs de convolution comparés dans l'expérience :

  1.  StandardConv       – Conv2d classique                          (baseline)
  2.  DeformableConv     – Conv déformable (Dai et al., 2017)
  3.  DynamicFilterConv  – Dynamic Filter Network (De Brabandere et al., 2016)
  4.  DynamicConv        – Dynamic Conv / attention sur K noyaux (Chen et al., 2020)
  5.  CondConv           – Conditionally Parameterized Conv (Yang et al., 2019)
  6.  PAC                – Pixel-Adaptive Conv (Su et al., 2019)
  7.  KNConv             – Kernel Normalized Conv (Nasirigerdeh et al., 2024)
  8.  HyperConv          – Hypernetwork-based Conv (Ha et al., 2017)
  9.  SAC                – Statistically Adaptive Conv (notre proposition)

Toutes les couches respectent la même interface :
    forward(x: Tensor[B,C,H,W]) -> Tensor[B,C_out,H',W']
afin de pouvoir être échangées sans modifier le détecteur.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONVOLUTION STANDARD
# ─────────────────────────────────────────────────────────────────────────────
class StandardConv(nn.Module):
    """Conv2d classique — filtre fixe spatialement invariant."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONV DÉFORMABLE  (Dai et al., ICCV 2017)
# ─────────────────────────────────────────────────────────────────────────────
class DeformableConv(nn.Module):
    """
    Convolution déformable : un réseau léger prédit des offsets 2D pour chaque
    position d'échantillonnage du noyau.  On implémente la version v1 sans
    modulation (pas de masque d'amplitude) pour rester simple.

    Ref: Dai et al., "Deformable Convolutional Networks", ICCV 2017.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, **kwargs):
        super().__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels

        # Réseau prédicateur d'offsets : 2 * k² offsets (Δx, Δy par position)
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=True
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # Prédit les offsets puis applique torchvision ou fallback manuel
        try:
            from torchvision.ops import deform_conv2d
            offset = self.offset_conv(x)          # (B, 2k², H', W')
            weight = self.conv.weight
            out = deform_conv2d(x, offset, weight,
                                stride=self.stride, padding=self.padding)
        except ImportError:
            # Fallback : convolution standard si torchvision non disponible
            out = self.conv(x)
        return self.act(self.bn(out))


# ─────────────────────────────────────────────────────────────────────────────
# 3. DYNAMIC FILTER NETWORK  (De Brabandere et al., NeurIPS 2016)
# ─────────────────────────────────────────────────────────────────────────────
class DynamicFilterConv(nn.Module):
    """
    Un réseau générateur (filter-generating network) produit un noyau unique
    par échantillon conditionné sur l'entrée globale (via global avg pool).
    Le noyau est ensuite appliqué en convolution dépthwise sur l'entrée.

    Ref: De Brabandere et al., "Dynamic Filter Networks", NeurIPS 2016.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, **kwargs):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

        # Filter-generating network : GAP → MLP → noyau de taille (out*in*k*k)
        self.filter_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_channels * in_channels * kernel_size * kernel_size)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        # Génère un noyau par image dans le batch
        filters = self.filter_gen(x)                   # (B, out*in*k*k)
        filters = filters.view(B * self.out_ch,
                               self.in_ch,
                               self.k, self.k)         # (B*out, in, k, k)
        # Applique le noyau image par image (grouped conv trick)
        x_grouped = x.view(1, B * C, H, W)             # (1, B*in, H, W)
        out = F.conv2d(x_grouped, filters,
                       stride=self.stride,
                       padding=self.padding,
                       groups=B)                        # (1, B*out, H', W')
        _, _, H2, W2 = out.shape
        out = out.view(B, self.out_ch, H2, W2)
        return self.act(self.bn(out))


# ─────────────────────────────────────────────────────────────────────────────
# 4. DYNAMIC CONVOLUTION  (Chen et al., CVPR 2020)
# ─────────────────────────────────────────────────────────────────────────────
class DynamicConv(nn.Module):
    """
    K noyaux statiques apprenables ; des poids d'attention calculés par GAP
    les combinent en un noyau effectif unique.

    Ref: Chen et al., "Dynamic Convolution: Attention over Convolution Kernels",
         CVPR 2020.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, num_kernels=4, **kwargs):
        super().__init__()
        self.K = num_kernels
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

        # K noyaux statiques
        self.kernels = nn.Parameter(
            torch.randn(num_kernels, out_channels, in_channels,
                        kernel_size, kernel_size) * 0.02
        )
        # Attention : GAP → FC → softmax
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_kernels),
            nn.Softmax(dim=1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B = x.shape[0]
        alpha = self.attention(x)                      # (B, K)
        # Combinaison pondérée des K noyaux
        # kernels : (K, out, in, k, k) → poids agrégé : (B, out, in, k, k)
        w = torch.einsum('bk,koihw->boihw', alpha, self.kernels)
        # Applique le noyau batch par batch
        outs = []
        for i in range(B):
            outs.append(F.conv2d(x[i:i+1], w[i],
                                 stride=self.stride, padding=self.padding))
        out = torch.cat(outs, dim=0)
        return self.act(self.bn(out))


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONDCONV  (Yang et al., NeurIPS 2019)
# ─────────────────────────────────────────────────────────────────────────────
class CondConv(nn.Module):
    """
    Identique à DynamicConv mais avec activation sigmoïde (pas softmax) sur les
    poids d'attention, et normalisation par leur somme.

    Ref: Yang et al., "CondConv: Conditionally Parameterized Convolutions",
         NeurIPS 2019.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, num_experts=4, **kwargs):
        super().__init__()
        self.K = num_experts
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernels = nn.Parameter(
            torch.randn(num_experts, out_channels, in_channels,
                        kernel_size, kernel_size) * 0.02
        )
        # Routing : sigmoid au lieu de softmax
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_experts),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B = x.shape[0]
        alpha = self.router(x)                         # (B, K)
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-6)
        w = torch.einsum('bk,koihw->boihw', alpha, self.kernels)
        outs = []
        for i in range(B):
            outs.append(F.conv2d(x[i:i+1], w[i],
                                 stride=self.stride, padding=self.padding))
        out = torch.cat(outs, dim=0)
        return self.act(self.bn(out))


# ─────────────────────────────────────────────────────────────────────────────
# 6. PAC — PIXEL-ADAPTIVE CONV  (Su et al., CVPR 2019)
# ─────────────────────────────────────────────────────────────────────────────
class PAC(nn.Module):
    """
    Le filtre fixe W est modulé par un noyau gaussien K calculé à partir de
    caractéristiques de guidage apprises.  On utilise l'entrée elle-même comme
    guidage (auto-PAC) pour rester sans branche extérieure.

        y_ij = Σ_s  W(s) · K(f_ij, f_s) · X_s
        K(f_i, f_j) = exp(-‖f_i - f_j‖² / (2σ²))

    Ref: Su et al., "Pixel-Adaptive Convolutional Neural Networks", CVPR 2019.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, guide_channels=8, **kwargs):
        super().__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        self.g = guide_channels

        # Branche de guidage : projette l'entrée en g canaux
        self.guide_conv = nn.Conv2d(in_channels, guide_channels, 1, bias=False)
        # Filtre de base fixe
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.sigma = nn.Parameter(torch.ones(1) * 0.5)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # Caractéristiques de guidage
        f = self.guide_conv(x)                         # (B, g, H, W)

        # Noyau adaptatif K par pixel : approximé par un scaling de l'entrée
        # (simplification tractable : on ne fait pas la double boucle sur voisins)
        # Approx : K_ij ≈ exp(-‖f_i‖² / 2σ²)  — per-pixel weighting
        norm_f = (f * f).sum(dim=1, keepdim=True)      # (B,1,H,W)
        K = torch.exp(-norm_f / (2 * self.sigma ** 2 + 1e-8))  # (B,1,H,W)

        # Modulation de l'entrée puis convolution
        x_mod = x * K
        out = self.conv(x_mod)
        return self.act(self.bn(out))


# ─────────────────────────────────────────────────────────────────────────────
# 7. KNCONV — KERNEL NORMALIZED CONV  (Nasirigerdeh et al., TMLR 2024)
# ─────────────────────────────────────────────────────────────────────────────
class KNConv(nn.Module):
    """
    Normalise le patch local (μ, σ du noyau k×k) avant la convolution standard.

        x̂_patch = (x_patch - μ_patch) / σ_patch
        y = W ⋆ x̂

    Ref: Nasirigerdeh et al., "Kernel Normalized Convolutional Networks",
         TMLR 2024.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, **kwargs):
        super().__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        k, s, p = self.k, self.stride, self.padding

        # Unfold : extrait tous les patches → (B, C*k*k, L) avec L = H'*W'
        patches = F.unfold(x, kernel_size=k, stride=s, padding=p)
        # patches : (B, C*k*k, L)

        # Normalisation par patch
        mu = patches.mean(dim=1, keepdim=True)         # (B, 1, L)
        std = patches.std(dim=1, keepdim=True) + 1e-6  # (B, 1, L)
        patches_norm = (patches - mu) / std            # (B, C*k*k, L)

        # Refold vers le tenseur spatial
        H_out = (H + 2*p - k) // s + 1
        W_out = (W + 2*p - k) // s + 1
        x_norm = F.fold(patches_norm,
                        output_size=(H_out * s, W_out * s),
                        kernel_size=k, stride=s, padding=p)
        # Correction du comptage (zones chevauchantes)
        ones = torch.ones_like(x)
        count = F.fold(F.unfold(ones, k, stride=s, padding=p),
                       (H_out * s, W_out * s), k, stride=s, padding=p)
        x_norm = x_norm / (count + 1e-8)
        # Redimensionne si nécessaire
        if x_norm.shape != x.shape:
            x_norm = x_norm[..., :H, :W]

        out = self.conv(x_norm)
        return self.act(self.bn(out))


# ─────────────────────────────────────────────────────────────────────────────
# 8. HYPERCONV  (Ha et al., ICLR 2017)
# ─────────────────────────────────────────────────────────────────────────────
class HyperConv(nn.Module):
    """
    Un hyperréseau (réseau secondaire) génère les poids du réseau primaire à
    partir d'un vecteur latent z appris.  Ici z est un embedding de taille fixe
    partagé (pas conditionné sur l'entrée — version statique de Ha et al.).

    Ref: Ha et al., "HyperNetworks", ICLR 2017.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, z_dim=16, **kwargs):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

        # Vecteur latent appris
        self.z = nn.Parameter(torch.randn(z_dim) * 0.01)

        # Hyperréseau : z → poids du réseau primaire
        self.hypernet = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels * in_channels * kernel_size * kernel_size)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # Génère les poids depuis z
        w = self.hypernet(self.z)
        w = w.view(self.out_ch, self.in_ch, self.k, self.k)
        out = F.conv2d(x, w, stride=self.stride, padding=self.padding)
        return self.act(self.bn(out))


# ─────────────────────────────────────────────────────────────────────────────
# 9. SAC — STATISTICALLY ADAPTIVE CONV  (notre proposition)
# ─────────────────────────────────────────────────────────────────────────────
class SAC(nn.Module):
    """
    Pour chaque position (i,j), calcule le vecteur de statistiques du patch :
        S_ij = [μ, σ, γ (skewness), κ (kurtosis), H (entropie)] ∈ R^5

    Un MLP partagé g_θ : R^5 → R^(C·k²) génère le noyau adaptatif W_ij.
    La convolution est alors :
        y_ij = <W_ij, P_ij>

    Notre proposition — voir le rapport technique SAC (2026).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, h_dim=64, n_bins=16, **kwargs):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_bins = n_bins

        # d = 5 descripteurs statistiques
        d = 5
        kernel_flat = in_channels * kernel_size * kernel_size

        # MLP générateur partagé g_θ : R^d → R^(out * in * k²)
        self.generator = nn.Sequential(
            nn.Linear(d, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, out_channels * kernel_flat)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    # ── Calcul des statistiques ───────────────────────────────────────────────
    @staticmethod
    @torch.no_grad()
    def _patch_stats(patches: torch.Tensor, n_bins: int) -> torch.Tensor:
        """
        patches : (B, C*k*k, L)  —  L = nombre de positions spatiales

        Retourne stats : (L_total, 5) avec L_total = B * L
        Les statistiques sont calculées sur la dimension C*k*k (les N=C·k² éléments
        du patch).  stop-gradient : @torch.no_grad().
        """
        B, N, L = patches.shape
        p = patches.permute(0, 2, 1).reshape(B * L, N).float()  # (B*L, N)

        mu = p.mean(dim=1)                             # (B*L,)
        diff = p - mu.unsqueeze(1)                     # (B*L, N)
        sigma = diff.pow(2).mean(dim=1).sqrt() + 1e-8  # (B*L,)

        z = diff / sigma.unsqueeze(1)                  # (B*L, N)  — standardisé
        skew = z.pow(3).mean(dim=1)                    # (B*L,)
        kurt = z.pow(4).mean(dim=1) - 3.0              # (B*L,)  — excès

        # Entropie locale : histogramme discret sur N valeurs
        p_min = p.min(dim=1, keepdim=True).values
        p_max = p.max(dim=1, keepdim=True).values
        p_norm = (p - p_min) / (p_max - p_min + 1e-8)  # (B*L, N) in [0,1]
        bin_idx = (p_norm * (n_bins - 1)).long().clamp(0, n_bins - 1)
        H = torch.zeros(B * L, device=p.device)
        for b in range(n_bins):
            prob = (bin_idx == b).float().mean(dim=1) + 1e-8
            H -= prob * prob.log()

        stats = torch.stack([mu, sigma, skew, kurt, H], dim=1)  # (B*L, 5)
        return stats

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k, s, p_pad = self.k, self.stride, self.padding

        H_out = (H + 2 * p_pad - k) // s + 1
        W_out = (W + 2 * p_pad - k) // s + 1
        L = H_out * W_out

        # Extrait tous les patches : (B, C*k*k, L)
        patches = F.unfold(x, kernel_size=k, stride=s, padding=p_pad)

        # Calcule les statistiques (stop-gradient)
        stats = self._patch_stats(patches, self.n_bins)    # (B*L, 5)
        stats = stats.to(x.dtype)

        # Génère les noyaux adaptatifs via le MLP partagé
        kernels = self.generator(stats)                    # (B*L, out*C*k²)
        kernels = kernels.view(B, L, self.out_ch,
                               self.in_ch * k * k)         # (B, L, out, C*k²)

        # Convolution adaptative position par position :
        # patches : (B, C*k*k, L) → (B, L, C*k*k, 1)
        p_t = patches.permute(0, 2, 1).unsqueeze(-1)      # (B, L, C*k², 1)

        # Produit scalaire : (B, L, out, C*k²) @ (B, L, C*k², 1) → (B, L, out, 1)
        out_flat = torch.matmul(kernels, p_t).squeeze(-1)  # (B, L, out)
        out_flat = out_flat.permute(0, 2, 1)               # (B, out, L)
        out = out_flat.view(B, self.out_ch, H_out, W_out)  # (B, out, H', W')

        return self.act(self.bn(out))


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRE
# ─────────────────────────────────────────────────────────────────────────────
CONV_REGISTRY = {
    "standard":       StandardConv,
    "deformable":     DeformableConv,
    "dynamic_filter": DynamicFilterConv,
    "dynamic_conv":   DynamicConv,
    "condconv":       CondConv,
    "pac":            PAC,
    "knconv":         KNConv,
    "hyperconv":      HyperConv,
    "sac":            SAC,
}


def build_conv(name: str, in_channels: int, out_channels: int, **kwargs):
    """Construit un opérateur de convolution par nom."""
    if name not in CONV_REGISTRY:
        raise ValueError(f"Convolution inconnue: '{name}'. "
                         f"Disponibles : {list(CONV_REGISTRY)}")
    return CONV_REGISTRY[name](in_channels, out_channels, **kwargs)
