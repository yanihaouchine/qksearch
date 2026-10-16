# qksearch — recherche automatique de circuits pour noyaux quantiques

*[English version](README.md)*

Un noyau quantique vaut ce que vaut sa *feature map*, et celle-ci est le plus
souvent dessinée à la main. Ce projet en fait une **recherche combinatoire** :
il construit des circuits IQP porte par porte et note chaque candidat par
l'**alignement noyau-cible** (KTA) de sa matrice de Gram, éventuellement
régularisé par la dimension de l'algèbre de Lie dynamique du circuit ou par la
dispersion du Gram. L'application visée est la régression de l'adsorption de
gaz dans des MOF (*metal-organic frameworks*).

Stage de recherche au Prasad Research Group, Université de Calgary (2026).


La liste complète des expériences est dans
[`docs/experiments.md`](docs/experiments.md).
