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
        try:
            from torchvision.ops import deform_conv2d
            offset = self.offset_conv(x)
            weight = self.conv.weight
            # deform_conv2d non supporté sur MPS → fallback CPU transparent
            if x.device.type == 'mps':
                out = deform_conv2d(
                    x.cpu(), offset.cpu(), weight.cpu(),
                    stride=self.stride, padding=self.padding
                ).to(x.device)
            else:
                out = deform_conv2d(x, offset, weight,
                                    stride=self.stride, padding=self.padding)
        except (ImportError, RuntimeError):
            # torchvision absent ou erreur inattendue → conv standard
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
# 9. SAC — STATISTICALLY ADAPTIVE CONV  (notre proposition — v2)
# ─────────────────────────────────────────────────────────────────────────────
class SAC(nn.Module):
    """
    Convolution Statistiquement Adaptative v2.

    Améliorations vs v1 :
    1. Résidu de base : SAC = Conv_standard(x) + delta_adaptatif(x)
       → la conv standard sert de point de départ solide,
         le MLP n'apprend que la CORRECTION adaptative.
       → convergence beaucoup plus rapide (le résidu peut commencer à zéro).

    2. MLP plus profond (3 couches) avec normalisation des stats
       → meilleure expressivité, gradients plus stables.

    3. Stats normalisées avant le MLP
       → évite les problèmes de scale entre μ (~0) et κ (~±2).

    Notre proposition — voir le rapport technique SAC (2026).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, h_dim=64, n_bins=8, **kwargs):
        super().__init__()
        self.in_ch   = in_channels
        self.out_ch  = out_channels
        self.k       = kernel_size
        self.stride  = stride
        self.padding = padding
        self.n_bins  = n_bins

        d           = 5
        kernel_flat = in_channels * kernel_size * kernel_size

        # ── Branche 1 : convolution de base (résidu) ─────────────────────
        # Initialise les poids normalement → convergence rapide dès l'epoch 1
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding, bias=False)

        # ── Branche 2 : correction adaptative via MLP ─────────────────────
        # Le MLP génère Δ_ij qui s'ajoute à la sortie de base_conv.
        # Initialisé proche de zéro → au départ SAC ≈ StandardConv.
        self.delta_gen = nn.Sequential(
            nn.Linear(d, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, out_channels * kernel_flat)
        )
        # Initialise la dernière couche à zéro → delta = 0 au départ
        nn.init.zeros_(self.delta_gen[-1].weight)
        nn.init.zeros_(self.delta_gen[-1].bias)

        # Scalaire appris pour contrôler l'amplitude de la correction
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    # ── Statistiques vectorisées ──────────────────────────────────────────
    @staticmethod
    @torch.no_grad()
    def _patch_stats(patches: torch.Tensor, n_bins: int) -> torch.Tensor:
        """
        [μ, σ, γ, κ, H] pour chaque patch — vectorisé GPU.
        patches : (BL, N)  →  stats : (BL, 5)  normalisées dans [-3, 3].
        """
        BL, N = patches.shape
        p = patches.float()

        mu    = p.mean(dim=1)
        diff  = p - mu.unsqueeze(1)
        sigma = diff.pow(2).mean(dim=1).sqrt() + 1e-8
        z     = diff / sigma.unsqueeze(1)
        skew  = z.pow(3).mean(dim=1)
        kurt  = z.pow(4).mean(dim=1) - 3.0

        # Entropie via scatter_add
        p_min   = p.min(dim=1, keepdim=True).values
        p_max   = p.max(dim=1, keepdim=True).values
        p_norm  = (p - p_min) / (p_max - p_min + 1e-8)
        bin_idx = (p_norm * (n_bins - 1)).long().clamp(0, n_bins - 1)
        counts  = torch.zeros(BL, n_bins, device=p.device)
        counts.scatter_add_(1, bin_idx, torch.ones(BL, N, device=p.device))
        probs   = counts / N + 1e-8
        H       = -(probs * probs.log()).sum(dim=1)

        stats = torch.stack([mu, sigma, skew, kurt, H], dim=1)  # (BL, 5)

        # Normalisation robuste par stats globales du batch
        # → évite que κ (±2) domine μ (~0) dans le MLP
        mean_s = stats.mean(dim=0, keepdim=True)
        std_s  = stats.std(dim=0, keepdim=True) + 1e-6
        stats  = (stats - mean_s) / std_s

        return stats.clamp(-5, 5)

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k, s, pad  = self.k, self.stride, self.padding

        # ── Branche 1 : résidu de base (conv standard) ────────────────────
        base = self.base_conv(x)             # (B, out, H', W')
        H_out, W_out = base.shape[2], base.shape[3]
        L = H_out * W_out
        N = self.in_ch * k * k

        # ── Extraction patches ────────────────────────────────────────────
        patches = F.unfold(x, kernel_size=k, stride=s, padding=pad)
        p_flat  = patches.permute(0, 2, 1).reshape(B * L, N)

        # ── Statistiques (stop-gradient) ─────────────────────────────────
        stats = self._patch_stats(p_flat, self.n_bins).to(x.dtype)

        # ── Génération chunked de la correction Δ ────────────────────────
        chunk = min(B * L, 4096)
        delta_list = []
        for start in range(0, B * L, chunk):
            end = min(start + chunk, B * L)
            delta_list.append(self.delta_gen(stats[start:end]))
        delta_kernels = torch.cat(delta_list, dim=0)            # (BL, out*N)
        delta_kernels = delta_kernels.reshape(B, L, self.out_ch, N)

        # ── Correction adaptative Δy ──────────────────────────────────────
        patches_t = p_flat.reshape(B, L, N)
        delta_y   = (delta_kernels * patches_t.unsqueeze(2)).sum(dim=-1)
        delta_y   = delta_y.permute(0, 2, 1).contiguous()
        delta_y   = delta_y.reshape(B, self.out_ch, H_out, W_out)

        # ── Sortie finale : base + α·Δ ────────────────────────────────────
        out = base + self.alpha * delta_y
        return self.act(self.bn(out))



# ─────────────────────────────────────────────────────────────────────────────
# 10. SAC_FAST — version rapide par groupes statistiques (notre proposition)
# ─────────────────────────────────────────────────────────────────────────────
class SAC_Fast(nn.Module):
    """
    SAC rapide par clustering statistique.

    Principe :
        Au lieu de générer un kernel par position (BL kernels, coûteux),
        on apprend K centroïdes dans l'espace statistique R^5.
        Chaque position est assignée (soft) à ses centroïdes les plus proches.
        On génère K kernels seulement, puis on combine.

    Complexité :
        SAC      : O(BL · h_dim · C·k²)   BL peut valoir 6400
        SAC_Fast : O(K  · h_dim · C·k²)   K = 8 typiquement
        → Gain : ~800× moins de passes MLP

    Hyperparamètre clé : n_clusters K
        K=4  → très rapide (~StandardConv speed), moins expressif
        K=8  → bon compromis vitesse/précision  (défaut)
        K=16 → proche de SAC plein mais bien plus rapide
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, h_dim=32, n_bins=8,
                 n_clusters=8, **kwargs):
        super().__init__()
        self.in_ch   = in_channels
        self.out_ch  = out_channels
        self.k       = kernel_size
        self.stride  = stride
        self.padding = padding
        self.n_bins  = n_bins
        self.K       = n_clusters

        d           = 5
        kernel_flat = in_channels * kernel_size * kernel_size

        # K centroïdes appris dans l'espace statistique R^d
        self.centroids = nn.Parameter(torch.randn(n_clusters, d) * 0.5)

        # MLP : génère K kernels depuis les K centroïdes (K passes seulement !)
        self.generator = nn.Sequential(
            nn.Linear(d, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, out_channels * kernel_flat)
        )

        # Température du soft-assignment (apprise)
        self.log_temp = nn.Parameter(torch.zeros(1))

        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    @torch.no_grad()
    def _patch_stats(patches: torch.Tensor, n_bins: int) -> torch.Tensor:
        BL, N = patches.shape
        p     = patches.float()
        mu    = p.mean(dim=1)
        diff  = p - mu.unsqueeze(1)
        sigma = diff.pow(2).mean(dim=1).sqrt() + 1e-8
        z     = diff / sigma.unsqueeze(1)
        skew  = z.pow(3).mean(dim=1)
        kurt  = z.pow(4).mean(dim=1) - 3.0
        p_min = p.min(dim=1, keepdim=True).values
        p_max = p.max(dim=1, keepdim=True).values
        p_n   = (p - p_min) / (p_max - p_min + 1e-8)
        bidx  = (p_n * (n_bins - 1)).long().clamp(0, n_bins - 1)
        cnt   = torch.zeros(BL, n_bins, device=p.device)
        cnt.scatter_add_(1, bidx, torch.ones(BL, N, device=p.device))
        prb   = cnt / N + 1e-8
        H     = -(prb * prb.log()).sum(dim=1)
        return torch.stack([mu, sigma, skew, kurt, H], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k, s, pad  = self.k, self.stride, self.padding
        H_out = (H + 2*pad - k) // s + 1
        W_out = (W + 2*pad - k) // s + 1
        L = H_out * W_out
        N = self.in_ch * k * k

        # ── 1. Patches et statistiques ────────────────────────────────────
        patches  = F.unfold(x, kernel_size=k, stride=s, padding=pad)
        p_flat   = patches.permute(0,2,1).reshape(B*L, N)
        stats    = self._patch_stats(p_flat, self.n_bins).to(x.dtype)  # (BL,5)

        # ── 2. Soft-assignment aux K centroïdes ───────────────────────────
        # dists : (BL, K) — distance L2 à chaque centroïde
        diff_c  = stats.unsqueeze(1) - self.centroids.unsqueeze(0)     # (BL,K,5)
        dists   = diff_c.pow(2).sum(dim=-1)                             # (BL,K)
        temp    = self.log_temp.exp().clamp(0.1, 10.0)
        weights = F.softmax(-dists / temp, dim=-1)                      # (BL,K)

        # ── 3. Génère K kernels depuis les K centroïdes ───────────────────
        # SEULEMENT K passes MLP au lieu de BL !
        kernels_K = self.generator(self.centroids)          # (K, out*N)
        kernels_K = kernels_K.reshape(self.K, self.out_ch, N)

        # ── 4. Kernel effectif par position = mélange pondéré ────────────
        # weights   : (BL, K)
        # kernels_K : (K, out, N)
        # →           (BL, out, N)
        kernels_eff = torch.einsum('bk,kon->bon', weights, kernels_K)
        kernels_eff = kernels_eff.reshape(B, L, self.out_ch, N)

        # ── 5. Produit scalaire adaptatif ─────────────────────────────────
        patches_t = p_flat.reshape(B, L, N).to(x.dtype)
        out_flat  = (kernels_eff * patches_t.unsqueeze(2)).sum(dim=-1)
        out = out_flat.permute(0,2,1).contiguous().reshape(
            B, self.out_ch, H_out, W_out)
        return self.act(self.bn(out))


# ─────────────────────────────────────────────────────────────────────────────
# 11. PWC — PROBABILITY-WEIGHTED CONVOLUTION  (notre nouvelle proposition)
# ─────────────────────────────────────────────────────────────────────────────
class PWC(nn.Module):
    """
    Probability-Weighted Convolution (PWC) — version optimisée.

    Pour chaque patch P_ij :
        1. Divise [p_min, p_max] en B=17 intervalles uniformes par patch
        2. Calcule la probabilité de chaque intervalle : p_b = count_b / N
        3. Poids de chaque pixel = 1 / (p_b(x) + ε)  → pixels rares amplifiés
        4. y_ij = <W, poids(P_ij)>   (W = filtre appris, identique partout)

    Optimisations :
        - aminmax() en une seule passe (au lieu de min + max séparés)
        - Bins calculés sur [p_min, p_max] réel (pas de normalisation [0,255])
        - Buffer ones pré-alloué et réutilisé entre les forward
        - matmul au lieu de einsum pour la conv (plus rapide sur CPU et GPU)

    Propriétés :
        - Zéro paramètre supplémentaire vs StandardConv
        - Différentiable par rapport à W (w traité en stop-gradient comme BN)
        - Amplifie automatiquement les petits objets rares dans chaque patch
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, n_bins=17, eps=1e-6, **kwargs):
        super().__init__()
        self.in_ch   = in_channels
        self.out_ch  = out_channels
        self.k       = kernel_size
        self.stride  = stride
        self.padding = padding
        self.n_bins  = n_bins
        self.eps     = eps

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.ReLU(inplace=True)

        # Buffer ones pré-alloué — évite une allocation par forward
        self._ones_cache: torch.Tensor = None
        self._ones_shape: tuple        = None

    @staticmethod
    @torch.no_grad()
    def _probability_weights(patches: torch.Tensor,
                             n_bins: int,
                             eps: float,
                             ones_cache=None) -> torch.Tensor:
        """
        patches : (B, N, L)  → retourne poids (B, N, L)
        """
        B, N, L = patches.shape
        BL = B * L

        # Reshape : une ligne par position spatiale
        p = patches.permute(0, 2, 1).reshape(BL, N).float()

        # Min/max en une seule passe
        p_min, p_max = p.aminmax(dim=1, keepdim=True)

        # Bin index sur [p_min, p_max]
        scale   = n_bins / (p_max - p_min + eps)
        bin_idx = ((p - p_min) * scale).long().clamp(0, n_bins - 1)

        # Histogramme par position
        counts = torch.zeros(BL, n_bins, device=p.device)
        ones   = ones_cache if (ones_cache is not None and
                                ones_cache.shape == (BL, N))                  else torch.ones(BL, N, device=p.device)
        counts.scatter_add_(1, bin_idx, ones)

        # Poids inverse + normalisation par la moyenne du patch
        inv_p = N / (counts + eps)
        w     = inv_p.gather(1, bin_idx)
        w     = w / (w.mean(dim=1, keepdim=True) + eps)

        return w.reshape(B, L, N).permute(0, 2, 1)   # (B, N, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k, s, pad  = self.k, self.stride, self.padding
        H_out = (H + 2*pad - k) // s + 1
        W_out = (W + 2*pad - k) // s + 1
        L  = H_out * W_out
        N  = C * k * k
        BL = B * L

        # Extraction des patches
        patches = F.unfold(x, kernel_size=k, stride=s, padding=pad)  # (B,N,L)

        # Buffer ones — mis à jour si la forme change
        if self._ones_shape != (BL, N):
            self._ones_cache = torch.ones(BL, N, device=x.device)
            self._ones_shape = (BL, N)

        # Poids (stop-gradient)
        weights = self._probability_weights(
            patches, self.n_bins, self.eps, self._ones_cache
        ).to(x.dtype)

        # Patch pondéré
        patches_w = patches * weights                          # (B, N, L)

        # Conv via matmul : W_flat @ patches_w → (B, out_ch, L)
        W_flat   = self.conv.weight.reshape(self.out_ch, N)
        out_flat = torch.matmul(W_flat, patches_w)
        out      = out_flat.reshape(B, self.out_ch, H_out, W_out)
        return self.act(self.bn(out))


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
    #"sac":            SAC,
    #"sac_fast":       SAC_Fast,
    "pwc":            PWC,
}


def build_conv(name: str, in_channels: int, out_channels: int, **kwargs):
    """Construit un opérateur de convolution par nom."""
    if name not in CONV_REGISTRY:
        raise ValueError(f"Convolution inconnue: '{name}'. "
                         f"Disponibles : {list(CONV_REGISTRY)}")
    return CONV_REGISTRY[name](in_channels, out_channels, **kwargs)
