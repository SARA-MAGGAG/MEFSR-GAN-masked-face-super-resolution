# Dataset — Masked CelebA-HQ Images

Le dataset utilisé pour ce projet n'est **pas hébergé sur ce dépôt GitHub** (volumineux). Il est disponible publiquement sur Kaggle :

🔗 **[masked-celebahq-images](https://www.kaggle.com/datasets/saramaggag/masked-celebahq-images)**

## Contenu

Le dataset contient des paires d'images issues de **CelebA-HQ**, avec des masques faciaux synthétiques appliqués, réparties comme suit :

| Split | Nombre de paires |
|---|---|
| Train | 19 742 |
| Validation | 2 467 |
| Test | 2 469 |
| **Total** | **24 678** |

- **Entrée :** image masquée, basse résolution (32×32)
- **Sortie attendue :** image restaurée, haute résolution (128×128) — super-résolution ×4

## Utilisation

1. Télécharge le dataset depuis le lien Kaggle ci-dessus (ou utilise-le directement dans un notebook Kaggle via `Add Data`).
2. Place les données selon la structure attendue par les notebooks dans `versions/*/` (voir chaque notebook pour le chemin exact utilisé, ex. `/kaggle/input/masked-celebahq-images/`).

## Génération des masques

Les masques faciaux synthétiques ont été appliqués sur les images originales de CelebA-HQ pour simuler le port d'un masque (occlusion de la partie basse du visage : nez, bouche, menton).
