# 🏠 Prédiction du Prix des Maisons – Techniques Avancées de Régression  
*Lilian Page, Torres Diego*
Ce projet a été réalisé avec l'aide de ChatGPT

## 🎯 Objectif du Projet

Ce projet vise à effectuer une **analyse complète de données** sur un dataset issu du concours Kaggle :  
👉 [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Le but est de comprendre les facteurs influençant le prix de vente des maisons et de construire un **modèle prédictif performant**, déployé via un **dashboard interactif**.

---

## 📥 Acquisition des Données

- **Source** : Kaggle – concours "House Prices: Advanced Regression Techniques".
- **Pourquoi ce choix ?** Ce dataset est riche, bien documenté et combine des variables numériques et catégorielles variées, ce qui en fait un excellent support pour une approche complète : nettoyage, EDA, feature engineering, modélisation et déploiement.

---

## 🗂 Description du Dataset

Le jeu de données décrit les caractéristiques détaillées de maisons résidentielles à Ames, Iowa (USA).  
Il comprend des informations sur la surface habitable, les matériaux, le quartier, la qualité de construction, la présence de garage, de piscine, etc.

---

## 🧼 Nettoyage et Préparation des Données

### Étapes réalisées :

1. **Suppression des colonnes non informatives** : `Street`, `Utilities` (valeurs constantes)
2. **Valeurs aberrantes supprimées** : `GrLivArea` > 4500
3. **Correction des erreurs temporelles** : suppression des dates incohérentes (> 2017)
4. **Gestion des valeurs manquantes numériques** :
   - Méthode contextuelle : `LotFrontage` imputé par médiane selon le quartier
   - Remplacement par 0 ou médiane selon la logique métier
5. **Variables numériques catégorielles transformées** : `MSSubClass`, `OverallCond`
6. **Valeurs manquantes catégorielles** : approche au cas par cas
7. **Traitement de l’asymétrie** : log(SalePrice), transformation Box-Cox
8. **Standardisation** : via `RobustScaler`

---

## 🔍 Analyse Exploratoire des Données (EDA)

- Calcul de statistiques descriptives
- Matrices de corrélation
- Étude des distributions des prix selon les quartiers, la qualité, etc.
- Identification de variables fortement liées à `SalePrice` (ex : `OverallQual`, `GrLivArea`)

---

## 📈 Visualisation des Données

Des visualisations claires ont été produites via :
- Diagrammes de dispersion (`GrLivArea` vs `SalePrice`, etc.)
- Boxplots (par quartier ou qualité)
- Heatmaps de corrélation
- Histogrammes avec transformation log

---

## 🧠 Feature Engineering

Création de nouvelles variables explicatives pour enrichir le modèle :

1. `IsGarage`, `IsFireplace`, `IsPool`, `IsSecondFloor`, `IsOpenPorch`, `IsWoodDeck`
2. `TotalSqrtFeet` = surface habitable + sous-sol
3. `TotalBaths` = somme pondérée des salles de bains
4. Recodage de `Neighborhood` en 3 classes socio-économiques
5. **Encodage One-Hot** pour toutes les variables catégorielles restantes

---

## 🤖 Modélisation (Machine Learning)

### Algorithmes testés :

- **Régressions linéaires régulières** : Lasso, ElasticNet  
- **Méthodes d’ensemble** : Gradient Boosting, XGBoost, LightGBM, Bagging  
- **Stacking** : StackingCVRegressor combinant plusieurs modèles

### Prédiction finale :

- **Combinaison pondérée** :
  - 20% ElasticNet  
  - 25% Lasso  
  - 15% LightGBM  
  - 40% Stacking (Lasso + ElasticNet + XGBoost + LightGBM)

---

## ✅ Évaluation du Modèle

- **Métrique utilisée** : RMSLE (Root Mean Squared Log Error)
- **Validation croisée** : k-fold (k=5) sur le dataset d’entraînement
- **Objectif atteint** : performance stable, généralisation assurée sur données de test

---

## 🚀 Déploiement – Dashboard Interactif

Un **dashboard Streamlit** a été développé pour :

- Explorer l’impact des variables
- Visualiser les prédictions en fonction des caractéristiques saisies

### Lancement :

python -m pip install --upgrade pip setuptools wheel
python -m pip install streamlit --force-reinstall
pip install plotly


```bash
streamlit run app.py
