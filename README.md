# Distillation ResNet50 -> ResNet18 sur CIFAR-100

Ce projet entraîne un modèle **ResNet-50 teacher** sur CIFAR-100, puis distille
ses connaissances vers un **ResNet-18 student**. Il contient les expériences
baseline et plusieurs méthodes de distillation :

- Student baseline sans distillation
- Knowledge Distillation classique, notée **KD**
- Attention Transfer, notée **AT**
- FitNet avec étape de *hint training* puis KD
- Decoupled Knowledge Distillation, notée **DKD**

Les commandes ci-dessous reprennent les journaux d'exécution conservés dans
`save/*.txt`. Le dossier `save/` est ignoré par Git parce qu'il contient les
checkpoints, logs, résultats de recherche et sorties Weights & Biases.

## Structure du projet

```text
config.py              Valeurs par défaut et petites grilles d'hyperparamètres
datasets/              Chargement CIFAR-100 et augmentations
distillation/          Pertes KD, DKD, AT, FitNet et hooks de features
models/                ResNet-18 et ResNet-50 adaptés aux images CIFAR 32x32
scripts/               Entraînements, grid search, recherches Optuna, résumés
utils/                 Checkpoints, métriques, Mixup, CutMix et boucles communes
Report/                Rapport final anonymisé du projet
environment.yml        Environnement Conda minimal
requirements.txt       Dépendances pip principales
```

Les dossiers générés localement ne doivent pas être versionnés :

```text
data/       Dataset CIFAR-100 téléchargé par torchvision
save/       Checkpoints, histories, summaries, courbes, résultats de recherche
wandb/      Logs locaux Weights & Biases
__pycache__/ Caches Python
```

Le rapport anonymisé du projet se trouve dans le dossier :

```text
Report/Rapport_anonymise.pdf
```


## Installation

Depuis la racine du projet :

```bash
conda env create -f environment.yml
conda activate distill_resnet
```

Pour CUDA, adaptez l'installation de PyTorch à votre GPU si nécessaire. Le
fichier `environment.yml` cible une installation Conda avec `pytorch-cuda`.

## Données et sorties

CIFAR-100 est téléchargé automatiquement au premier lancement dans `data/`.
Les sorties suivent cette convention :

```text
save/<type_experience>/<modele>/<nom_experience>/
```

Chaque expérience peut produire :

```text
checkpoint.pth              Dernier checkpoint
best.pth                    Meilleur checkpoint
history.json                Historique des pertes et accuracies
summary.json                Résumé final
progress.json               État courant ou terminal
run_config.json             Configuration de l'exécution
loss_curve.png              Courbe de loss
accuracy_curve.png          Courbe d'accuracy
confusion_matrix.pt         Matrice de confusion
roc_metrics.json            AUC micro/macro
roc_curve_micro_macro.png   Courbe ROC
class_metrics.json          Métriques par classe
test_metrics.json           Métriques test globales
```

## Utilisation générale

Toutes les commandes doivent être lancées depuis la racine du projet avec
`python -m ...`.

Sélection GPU :

```bash
--gpu 0
```

Mode avec validation : ne pas passer `--full_train`. Les scripts utilisent alors
45 000 images train, 5 000 validation et 10 000 test.

Mode final : passer `--full_train`. Les scripts utilisent alors les 50 000 images
train et évaluent sur le test set.

Weights & Biases est optionnel. Pour désactiver W&B, retirez `--use_wandb` ou
utilisez :

```bash
--wandb_mode disabled
```

Pour activer W&B :

```bash
wandb login
```

Lorsque l'argument `--wandb_entity` est présent, remplacez `votre-entite-wandb`
par votre propre entité Weights & Biases, ou retirez l'argument si vous utilisez
l'entité par défaut de votre compte.

## Ordre recommandé des expériences

1. Rechercher les hyperparamètres du teacher.
2. Entraîner le teacher final ResNet-50.
3. Rechercher puis entraîner le student baseline ResNet-18.
4. Rechercher puis entraîner KD, AT, FitNet et DKD en utilisant le checkpoint teacher.
5. Comparer les checkpoints finaux.

Le checkpoint teacher attendu par les scripts de distillation est :

```text
save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth
```

## Recherche Teacher

Grid search rapide du teacher :

```bash
python -m scripts.search_teacher_grid \
  --model resnet50_cifar \
  --gpu 0 \
  --epochs 40 \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_tags teacher,grid,resnet50,cifar100 \
  --wandb_mode online \
  --campaign_id v1
```

Recherche Optuna du teacher :

```bash
python -m scripts.search_teacher_optuna \
  --model resnet50_cifar \
  --gpu 0 \
  --epochs 40 \
  --n_trials 12 \
  --search_preset quick \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_tags teacher,search,optuna,resnet50,cifar100 \
  --wandb_mode online \
  --campaign_id v1
```

## Entraînement teacher final

Commande finale issue de la meilleure recherche Optuna :

```bash
python -m scripts.train_teacher \
  --model resnet50_cifar \
  --gpu 0 \
  --epochs 240 \
  --full_train \
  --exp_name teacher_final_optuna_best \
  --lr 0.05 \
  --wd 0.001 \
  --label_smoothing 0.1 \
  --scheduler cosine \
  --mixup_alpha 0.4 \
  --no_cutmix \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_tags teacher,final,optuna,resnet50,cifar100 \
  --wandb_mode online \
  --wandb_group teacher-final/resnet50_cifar \
  --wandb_job_type final \
  --wandb_run_name final_resnet50_cifar_optuna_best \
  --wandb_log_artifacts
```

Sortie principale :

```text
save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth
```

## Recherche student baseline

Grid search rapide du student :

```bash
python -m scripts.search_student_grid \
  --model resnet18_cifar \
  --gpu 0 \
  --epochs 40 \
  --grid_preset quick \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_tags student,search,grid,resnet18,cifar100 \
  --wandb_mode online \
  --campaign_id v1
```

Recherche Optuna du student :

```bash
python -m scripts.search_student_optuna \
  --model resnet18_cifar \
  --gpu 0 \
  --epochs 40 \
  --n_trials 12 \
  --search_preset quick \
  --study_name student_optuna_quick \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_tags student,search,optuna,resnet18,cifar100 \
  --wandb_mode online \
  --campaign_id v1
```

## Entraînement student baseline final

```bash
python -m scripts.train_student \
  --model resnet18_cifar \
  --gpu 0 \
  --epochs 240 \
  --full_train \
  --exp_name student_final_baseline \
  --lr 0.08 \
  --wd 0.0005 \
  --label_smoothing 0.1 \
  --scheduler cosine \
  --mixup_alpha 0.2 \
  --cutmix_alpha 1.0 \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_tags student,final,baseline,resnet18,cifar100 \
  --wandb_mode online \
  --wandb_group student-final/resnet18_cifar \
  --wandb_job_type final \
  --wandb_run_name final_resnet18_cifar_baseline \
  --wandb_log_artifacts
```

Sortie principale :

```text
save/student/resnet18_cifar/student_final_baseline/best.pth
```

## Recherche KD

Grid search KD :

```bash
python -m scripts.search_kd_grid \
  --teacher_model resnet50_cifar \
  --student_model resnet18_cifar \
  --teacher_checkpoint ./save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth \
  --gpu 0 \
  --epochs 40 \
  --grid_preset quick \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_tags kd,search,grid,resnet50,resnet18,cifar100 \
  --wandb_mode online \
  --campaign_id v1
```

Recherche Optuna KD :

```bash
python -m scripts.search_kd_optuna \
  --teacher_model resnet50_cifar \
  --student_model resnet18_cifar \
  --teacher_checkpoint ./save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth \
  --gpu 0 \
  --epochs 40 \
  --n_trials 12 \
  --search_preset quick \
  --study_name kd_optuna \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_tags kd,search,optuna,resnet50,resnet18,cifar100 \
  --wandb_mode online \
  --campaign_id v1
```

## Entraînement KD final

```bash
python -m scripts.train_kd \
  --teacher_model resnet50_cifar \
  --student_model resnet18_cifar \
  --teacher_checkpoint ./save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth \
  --gpu 0 \
  --epochs 240 \
  --full_train \
  --exp_name kd_final_best \
  --lr 0.08 \
  --wd 0.0005 \
  --kd_temperature 2.0 \
  --kd_alpha 0.9 \
  --hard_label_smoothing 0.0 \
  --mixup_alpha 0.2 \
  --cutmix_alpha 1.0 \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_tags kd,final,resnet50,resnet18,cifar100 \
  --wandb_mode online \
  --wandb_group kd-final/resnet18_cifar \
  --wandb_job_type final \
  --wandb_run_name final_kd_resnet18_from_resnet50 \
  --wandb_log_artifacts
```

Sortie principale :

```text
save/kd/resnet18_cifar/kd_final_best/best.pth
```

## Recherche AT

Grid search Attention Transfer :

```bash
python -m scripts.search_at_grid \
  --teacher_model resnet50_cifar \
  --student_model resnet18_cifar \
  --teacher_checkpoint ./save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth \
  --gpu 0 \
  --epochs 40 \
  --grid_preset quick \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_tags at,search,grid,resnet50,resnet18,cifar100 \
  --wandb_mode online \
  --campaign_id v1
```

Recherche Optuna Attention Transfer :

```bash
python -m scripts.search_at_optuna \
  --teacher_model resnet50_cifar \
  --student_model resnet18_cifar \
  --teacher_checkpoint ./save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth \
  --gpu 0 \
  --epochs 40 \
  --n_trials 12 \
  --search_preset quick \
  --study_name at_optuna \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_tags at,search,optuna,resnet50,resnet18,cifar100 \
  --wandb_mode online \
  --campaign_id v1
```

## Entraînement AT final

```bash
python -m scripts.train_at \
  --teacher_model resnet50_cifar \
  --student_model resnet18_cifar \
  --teacher_checkpoint ./save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth \
  --gpu 0 \
  --epochs 240 \
  --full_train \
  --exp_name at_final_best \
  --lr 0.08 \
  --wd 0.0005 \
  --at_beta 50.0 \
  --hard_label_smoothing 0.0 \
  --teacher_layers layer2.1,layer3.1 \
  --student_layers layer2.1,layer3.1 \
  --mixup_alpha 0.2 \
  --cutmix_alpha 1.0 \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_tags at,final,resnet50,resnet18,cifar100 \
  --wandb_mode online \
  --wandb_group at-final/resnet18_cifar \
  --wandb_job_type final \
  --wandb_run_name final_at_resnet18_from_resnet50 \
  --wandb_log_artifacts
```

Sortie principale :

```text
save/at/resnet18_cifar/at_final_best/best.pth
```

## Recherche FitNet

Grid search FitNet :

```bash
python -m scripts.search_fitnet_grid \
  --teacher_model resnet50_cifar \
  --student_model resnet18_cifar \
  --teacher_checkpoint save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth \
  --gpu 0 \
  --epochs 20 \
  --hint_epochs 20 \
  --max_trials 8 \
  --save_subdir fitnet_grid \
  --campaign_id paper_v1 \
  --grid_preset quick \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_entity votre-entite-wandb \
  --wandb_tags fitnet,search,grid,cifar100 \
  --wandb_mode online
```

Recherche Optuna FitNet :

```bash
python -m scripts.search_fitnet_optuna \
  --teacher_model resnet50_cifar \
  --student_model resnet18_cifar \
  --teacher_checkpoint save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth \
  --gpu 0 \
  --epochs 20 \
  --hint_epochs 20 \
  --n_trials 12 \
  --study_name fitnet_optuna_tight \
  --save_subdir fitnet_optuna \
  --campaign_id paper_v1 \
  --search_preset quick \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_entity votre-entite-wandb \
  --wandb_tags fitnet,search,optuna,cifar100 \
  --wandb_mode online
```

## Entraînement FitNet final

FitNet utilise deux phases :

- `hint_epochs` : apprentissage du régresseur de features.
- `stage2_epochs` : distillation finale KD sur les logits.

```bash
python -m scripts.train_fitnet \
  --teacher_model resnet50_cifar \
  --student_model resnet18_cifar \
  --teacher_checkpoint save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth \
  --gpu 0 \
  --full_train \
  --exp_name fitnet_final_best_fulltrain \
  --save_subdir fitnet_final \
  --hint_epochs 60 \
  --stage2_epochs 240 \
  --hint_lr 5e-4 \
  --stage2_lr 0.08 \
  --wd 5e-4 \
  --kd_temperature 3.0 \
  --kd_alpha 0.9 \
  --hard_label_smoothing 0.0 \
  --teacher_hint_layers layer3.1 \
  --student_guided_layers layer2.1 \
  --mixup_alpha 0.2 \
  --cutmix_alpha 1.0 \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_entity votre-entite-wandb \
  --wandb_tags fitnet,final,fulltrain,paper,cifar100 \
  --wandb_mode online \
  --wandb_group fitnet/resnet18_cifar/final \
  --wandb_job_type train \
  --wandb_run_name fitnet_final_best_fulltrain
```

Sortie principale :

```text
save/fitnet_final/resnet18_cifar/fitnet_final_best_fulltrain/best.pth
```

## Recherche DKD

Grid search DKD :

```bash
python -m scripts.search_dkd_grid \
  --teacher_model resnet50_cifar \
  --student_model resnet18_cifar \
  --teacher_checkpoint save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth \
  --gpu 0 \
  --epochs 20 \
  --max_trials 8 \
  --save_subdir dkd_grid \
  --campaign_id paper_v1 \
  --grid_preset quick \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_entity votre-entite-wandb \
  --wandb_tags dkd,search,grid,cifar100 \
  --wandb_mode online
```

Recherche Optuna DKD :

```bash
python -m scripts.search_dkd_optuna \
  --teacher_model resnet50_cifar \
  --student_model resnet18_cifar \
  --teacher_checkpoint save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth \
  --gpu 0 \
  --epochs 20 \
  --n_trials 12 \
  --study_name dkd_optuna_tight \
  --save_subdir dkd_optuna \
  --campaign_id paper_v1 \
  --search_preset quick \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_entity votre-entite-wandb \
  --wandb_tags dkd,search,optuna,cifar100 \
  --wandb_mode online
```

## Entraînement DKD final

```bash
python -m scripts.train_dkd \
  --teacher_model resnet50_cifar \
  --student_model resnet18_cifar \
  --teacher_checkpoint save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth \
  --gpu 0 \
  --full_train \
  --exp_name dkd_final_optuna_best_fulltrain \
  --save_subdir dkd_final \
  --epochs 240 \
  --lr 0.08 \
  --wd 5e-4 \
  --dkd_temperature 2.0 \
  --dkd_alpha 0.5 \
  --dkd_beta 12.0 \
  --dkd_warmup_epochs 20 \
  --hard_label_smoothing 0.0 \
  --mixup_alpha 0.2 \
  --cutmix_alpha 1.0 \
  --use_wandb \
  --wandb_project distill_cifar100 \
  --wandb_entity votre-entite-wandb \
  --wandb_tags dkd,final,fulltrain,optuna,cifar100 \
  --wandb_mode online \
  --wandb_group dkd/resnet18_cifar/final \
  --wandb_job_type train \
  --wandb_run_name dkd_final_optuna_best_fulltrain
```

Sortie principale :

```text
save/dkd_final/resnet18_cifar/dkd_final_optuna_best_fulltrain/best.pth
```

## Reprise d'entraînement

Les scripts d'entraînement acceptent `--resume` :

```bash
python -m scripts.train_kd \
  --teacher_checkpoint save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth \
  --resume save/kd/resnet18_cifar/kd_final_best/checkpoint.pth
```

Conservez les autres arguments identiques à l'exécution initiale pour éviter de
mélanger plusieurs configurations dans un même dossier.

## Évaluation et comparaison

Les entraînements finaux évaluent le test set sauf si `--skip_test_metrics` est
utilisé. Pour comparer le nombre de paramètres, la taille théorique des poids
en float32 et la taille des fichiers checkpoint :

```bash
python -m scripts.summarize_checkpoints --save_dir save
```

Les checkpoints attendus par ce script sont :

```text
save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth
save/student/resnet18_cifar/student_final_baseline/best.pth
save/kd/resnet18_cifar/kd_final_best/best.pth
save/at/resnet18_cifar/at_final_best/best.pth
save/fitnet_final/resnet18_cifar/fitnet_final_best_fulltrain/best.pth
save/dkd_final/resnet18_cifar/dkd_final_optuna_best_fulltrain/best.pth
```

La colonne `Weights f32 (MB)` correspond à la comparaison de compacité du
rapport. La colonne `Checkpoint (MiB)` correspond au fichier sauvegardé sur
disque, qui peut être plus gros car il contient aussi des informations de run.

## Commandes de debug rapides

Lancer une expérience courte sans W&B :

```bash
python -m scripts.train_student \
  --model resnet18_cifar \
  --gpu 0 \
  --epochs 2 \
  --exp_name debug_student \
  --save_subdir debug \
  --wandb_mode disabled \
  --skip_test_metrics \
  --skip_plots
```

Tester une recherche sur une seule configuration :

```bash
python -m scripts.search_kd_grid \
  --teacher_model resnet50_cifar \
  --student_model resnet18_cifar \
  --teacher_checkpoint save/teacher/resnet50_cifar/teacher_final_optuna_best/best.pth \
  --gpu 0 \
  --epochs 2 \
  --max_trials 1 \
  --wandb_mode disabled
```

## Reproductibilité

- Le seed par défaut est défini dans `config.py`.
- `--benchmark` active cuDNN benchmark pour accélérer l'entraînement, avec une
  reproductibilité moins stricte.
- `--allow_tf32` active TF32 sur les GPU NVIDIA compatibles.
- `--full_train` supprime la validation et entraîne sur les 50 000 images train.
- Les recherches utilisent le split train/validation standard du projet.

## Journal des commandes historiques

Les commandes utilisées pour construire les expériences originales sont
conservées localement dans :

```text
save/grid_search_teacher.txt
save/optuna_search_teacher.txt
save/search_grid_student.txt
save/search_optuna_student.txt
save/search_kd_grid.txt
save/search_kd_optuna.txt
save/search_at_grid.txt
save/search_at_optuna.txt
save/search_fitnet_grid.txt
save/search_fitnet_optuna.txt
save/search_dkd_grid.txt
save/search_dkd_optuna.txt
save/train_teacher.txt
save/train_student_baseline.txt
save/train_kd.txt
save/train_at.txt
save/train_fitnet.txt
save/train_dkd.txt
```
