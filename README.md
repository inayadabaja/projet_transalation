# projet_transalation

---

Merci pour le lien vers ton dépôt GitHub ! J'ai consulté le projet et voici une version améliorée du README, adaptée à ton application Gradio de traduction et de visualisation, incluant des sections détaillées et une structure claire.

---

# 🌐 Projet de Traduction Automatique avec Visualisation

## 📘 Description

Cette application interactive, développée avec Gradio, permet de :

* Traduire du texte entre l’anglais et le français en utilisant plusieurs modèles de traduction.
* Visualiser les performances des modèles selon différentes métriques et paramètres.
* Explorer des exemples concrets de traductions générées par les modèles.

L'interface est moderne et personnalisée pour offrir une expérience utilisateur agréable et intuitive.

---

## 🚀 Fonctionnalités

* **Traduction en temps réel** : Traduisez instantanément du texte de l’anglais vers le français et vice versa.
* **Choix de modèles variés** : Sélectionnez parmi plusieurs modèles de traduction disponibles.
* **Ajustement de la température** : Modifiez la température pour contrôler la créativité des traductions.
* **Visualisation des performances** : Analysez les performances des modèles selon des métriques telles que BLEU et ROUGE.
* **Affichage d’exemples de traductions** : Consultez des exemples de traductions générées par les modèles sélectionnés.
* **Interface utilisateur intuitive** : Profitez d'une interface moderne et facile à utiliser, conçue avec soin.

---

## 🛠️ Installation

### Prérequis

* Python 3.8 ou supérieur
* pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Cloner le dépôt** :

   ```bash
   git clone https://github.com/inayadabaja/projet_transalation.git
   cd projet_transalation
   ```

2. **Créer un environnement virtuel** (optionnel mais recommandé) :

   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```

3. **Installer les dépendances** :

   ```bash
   pip install -r requirements.txt
   ```

4. **Lancer l'application** :

   ```bash
   python app.py
   ```

   L'application sera accessible à l'adresse suivante : [http://localhost:7860](http://localhost:7860)

---


### 📥 Source du Dataset

Le dataset utilisé pour cette démonstration est disponible sur Kaggle :
👉 [English-French Translation Dataset](https://www.kaggle.com/datasets/adewoleakorede/english-french-translation)

Il contient des paires de phrases en anglais et en français, utiles pour entraîner et tester des modèles de traduction automatique.

---

### 🧼 Étapes de Lecture et de Nettoyage

Le fichier d’origine présente des problèmes d’encodage typiques avec les caractères accentués en français. Voici les principales étapes suivies pour le rendre exploitable :

1. **Chargement du fichier CSV** avec le bon encodage (`latin1`) pour éviter les erreurs `UnicodeDecodeError`.
2. **Nettoyage des colonnes** inutiles (une colonne vide était présente).
3. **Correction des caractères mal encodés** (ex. `Ã©` remplacé par `é`) à l’aide d’un outil de réparation automatique.
4. **Suppression des lignes incomplètes** ou contenant des valeurs manquantes.
5. **Extraction d’un échantillon aléatoire** pour tester les modèles de traduction dans la démo Gradio.

Le fichier propre généré à la fin de ce processus est enregistré sous le nom `sample_fr_en_clean.csv` et utilisé pour alimenter l'application de traduction.

---

## 📂 Structure du projet

```plaintext
projet_transalation/
├── app.py                    # Script principal de l'application Gradio
├── comprehensive_evaluation.py
├── config.py
├── data_preprocessing.py
├── setup.sh
├── translation_demo.py
├── utils.py
├── visualiser.py
├── requirements.txt          # Liste des dépendances Python
├── results/                  # Dossier contenant les résultats de traduction
│   ├── comprehensive_results.json  # Résultats détaillés des traductions
│   └── evaluation_summary.json    # Résumé des performances des modèles
└── README.md                 # Documentation du projet

```

---

## ⚙️ Configuration

L'application utilise un fichier de configuration interne (`GRADIO_CONFIG`) pour définir :

* **Thème graphique** : Personnalisation de l'apparence de l'interface.
* **Paramètres du serveur** : Définition du nom du serveur, du port et de l'option de partage.
* **Modèles disponibles** : Liste des modèles de traduction accessibles dans l'application.

Vous pouvez ajuster ces paramètres selon vos préférences dans le fichier `app.py`.

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour proposer des améliorations, des corrections ou de nouvelles fonctionnalités :

1. Forkez ce dépôt.
2. Créez une branche dédiée à votre fonctionnalité : `git checkout -b feature/ma-fonctionnalite`.
3. Faites vos modifications et testez-les.
4. Soumettez une Pull Request en expliquant clairement les changements.

---

## 📞 Contact

Pour toute question, suggestion ou demande d’aide, vous pouvez me contacter via :

* [GitHub Issues](https://github.com/inayadabaja/projet_transalation/issues)
---

## 🎨 Remerciements

Merci aux communautés **Gradio**, **Hugging Face** et **Ollama** pour leurs outils et ressources qui rendent ce projet possible.

---

Souhaites-tu également que je t'aide à créer le fichier `requirements.txt` ou à préparer les fichiers JSON nécessaires pour les résultats de traduction ?


