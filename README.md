**TIPE MPI 2026 : Comment un bipède ne connaissant rien finit par savoir marcher ?**

https://github.com/user-attachments/assets/b971f95d-df1c-479e-a042-8050308159a9

Ce problème a été résolu **par apprentissage par renforcement** où le but était de trouver une politique capable de prédire les bonnes rotations d’articulation du bipède à  partir de l’état du système d’avant afin qu’il puisse marcher. Le projet regroupe plusieurs tentatives d’algorithmes : 
- **Reinforce**
- **A2C**
- **PPO**
C’est la méthode PPO (Proximal Policy Optimization) qui a fini par fonctionner


**Bibliothèques utilisées** : gymnasium (attention à la version), torch, matplotlib, numpy
