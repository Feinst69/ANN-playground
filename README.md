# ANN-playground

# :poop: :bug: :bulb: :lipstick: :bento: :coffin:

## Veille Technique


➔ Réalisez une veille sur le réseau de neurones artificiels de type
Perceptron Multicouches et expliquez son architecture (couches
d’entrée, couches cachées, couches de sortie, etc).


Le perceptron multicouche (PMC) est un réseau neuronal organisé en plusieurs couches au sein desquelles l'information circule de la couche d’entrée vers la couche de sortie (propagation avant).
Il est composé de trois "sections" traversées dans cet ordre, la couche d'entrée, les couches cachées et la couche de sortie.
Chaque couche est constituée d’un nombre variable de neurones
-----


➔ Expliquez le choix de l’architecture du PMC en fonction de la
problématique de classement ou de régression.

L'architecture d'un PMC va différer au niveaux de la fonction d'activation associée à la couche de sortie selon s'il s'agit d'une tâche de régression ou de classification, une tâche de régression aura une fonction d'activation de type linéaire , tandis que celle d'une classification binaire sera sigmoïde  et celle d'une classification multiclasse sera une fonction d'activation dîtes softmax



-----
➔ Définissez les termes suivants : Fonction d’activation, Propagation,
rétropropagation, Loss-function, Descente de gradient, Vanishing
gradients.




Définitions des termes :
1. Fonction d’activation
La fonction d'activation est une fonction mathématique appliquée à un signal en sortie d'un neurone artificiel

Exemple : Sigmoïde, ReLU, Tanh, Softmax.

2. Propagation (Forward Propagation)
Dans le neurone physique, l'influx nerveux circule toujours dans le même sens, c'est ce qu'on entend ici par propagation, les données traversent le réseaux de neurones dans un seul et même sens, de la couche d'entrée à la couche de sortie.

À chaque couche, les entrées sont transformées via des poids, biais et fonctions d’activation pour produire une sortie.
C’est ainsi que le réseau génère des prédictions ou valeurs intermédiaires pour une donnée donnée.



3. Rétropropagation (Backpropagation)
La rétropropagation est l’algorithme d’apprentissage utilisé pour ajuster les poids et les biais dans un réseau de neurones.

Elle repose sur la différenciation de la fonction de perte (loss function) par rapport aux paramètres du modèle (poids et biais).
Le gradient calculé est utilisé pour mettre à jour les paramètres dans le sens opposé à celui de l’erreur afin de réduire la loss.
Elle fonctionne en deux étapes :
Calcul des gradients en utilisant le backward pass, c'est à dire mesurer l'erreur  qui va servir a ajuster les pondérations et les biais.
Mise à jour des paramètres via un optimiseur (descente de gradient le plus souvent).

4. Loss Function (Fonction de perte)
La fonction de perte mesure l’écart entre les prédictions du modèle et les valeurs réelles (vérités terrain).

Le choix de la loss function est important dans la mesure ou c'est la métrique qui va être est utilisée pour guider l’apprentissage du modèle.
Exemples :
MSE (Mean Squared Error) pour les régressions.
Binary Cross-Entropy pour les classifications binaires.
Categorical Cross-Entropy pour les classifications multiclasse.

5. Descente de gradient (Gradient Descent)
La descente de gradient est une méthode d’optimisation utilisée pour ajuster les hyperparamètres d’un réseau de neurones en minimisant la fonction de perte.

À chaque étape, les poids et biais sont mis à jour en soustrayant une fraction (le taux d’apprentissage) du gradient de la perte.
Variantes courantes :
Batch Gradient Descent : Utilise toutes les données pour chaque mise à jour.
Stochastic Gradient Descent (SGD) : Met à jour les paramètres après chaque exemple.
Mini-Batch Gradient Descent : Combine les deux précédents en utilisant un sous-ensemble de données.

6. Vanishing Gradients
Le problème de la disparition de gradients  (vanishing gradients) survient lorsqu’au cours de la rétropropagation, les gradients deviennent très petits (proches de zéro) dans les couches profondes du réseau, rendant l'entraînement des réseaux très profonds particulièrement difficile.
Cela se produit souvent avec des fonctions d’activation comme sigmoïde ou tanh.

Les solutions pour paliers ce problème sont multiples:
- Utiliser des fonctions d’activation comme ReLU.
- Initialisation appropriée des poids.
- Architectures comme les réseaux résiduels (ResNets).
Ces concepts sont essentiels pour comprendre le fonctionnement des réseaux de neurones et les défis rencontrés lors de leur entraînement.

-----
➔ Présentez au moins 5 hyper-paramètres d’un réseau de neurones et
donnez les bonnes pratiques en termes de choix des valeurs.

- Batch size
- choix de la fonction d'activation
- Poids initiaux du réseaux de neurones
- Dropout
- Learning rate
- nombre de couches cachées
- optimizer


1. Batch-size (Taille du lot)
Le batch-size détermine combien d'exemples sont traités avant qu'une mise à jour des poids du modèle ne soit effectuée pendant l'entraînement.

Bonnes pratiques :
Petits batchs (ex. : 32, 64) :
Avantages : Plus grande capacité de généralisation, meilleur échantillonnage des gradients. Cela introduit également un certain bruit dans l'optimisation, ce qui peut aider à éviter le sur-apprentissage.
Inconvénients : Nécessite plus d'itérations pour chaque époque, ce qui peut augmenter le temps d'entraînement global.
Grands batchs (ex. : 128, 256, 512) :
Avantages : Calcul plus rapide grâce à une meilleure parallélisation, et plus de stabilité dans la mise à jour des poids.
Inconvénients : Peut provoquer un sur-apprentissage, car une taille de lot plus grande peut rendre l'optimisation plus déterministe et moins "exploratoire", ce qui pourrait nuire à la généralisation.
Pratique générale :
32 à 64 pour la plupart des tâches classiques (classification, régression), en fonction de la mémoire GPU disponible.
Si vous avez plus de ressources, vous pouvez tester des tailles de batch plus grandes, mais surveillez les performances.


2. Choix de la fonction d'activation
Les fonctions d'activation déterminent la manière dont les entrées aux neurones sont transformées en sorties, ce qui influe directement sur la capacité du réseau à apprendre des relations complexes.

Bonnes pratiques :
ReLU (Rectified Linear Unit) :

Avantages : Très populaire pour ses bonnes performances dans les réseaux profonds. Elle permet d'éviter la saturation des gradients et facilite l'entraînement.
Inconvénients : Risque de "neurones morts" si une grande partie des neurones n'active jamais (problème de vanishing gradient).
Leaky ReLU :

Avantages : Une version modifiée de ReLU qui permet de passer un petit gradient même lorsque l'entrée est négative, évitant ainsi le problème des neurones morts.
Sigmoid :

Avantages : Peut être utilisé pour des tâches de classification binaire.
Inconvénients : Saturation dans les extrêmes, ce qui peut ralentir l'apprentissage dans des réseaux profonds (problème de vanishing gradient).
Softmax :

Avantages : Utilisée principalement dans la couche de sortie pour les problèmes de classification multi-classes, car elle permet de transformer les scores en probabilités.
Pratique générale :
Pour des couches cachées, ReLU est souvent le meilleur choix.
Pour la classification binaire, la fonction d'activation de la couche de sortie doit être sigmoid.
Pour la classification multi-classes, utilisez softmax dans la couche de sortie.


3. Poids initiaux du réseau de neurones
Les poids initiaux déterminent comment les neurones du modèle sont initialisés avant l'entraînement.

Bonnes pratiques :
Initialisation aléatoire :

Avantages : Favorise une meilleure exploration du processus d'optimisation.
Exemples : glorot_uniform ou he_normal pour ReLU. Ces initialisations ajustent les poids en fonction du nombre de neurones dans les couches précédentes.
Initialisation de Xavier/Glorot (pour les réseaux avec des activations sigmoid ou tanh) :

Avantages : L'initialisation de Xavier permet d'éviter les problèmes de gradients trop grands ou trop petits, en maintenant la variance des activations similaire à travers les couches.
Initialisation de He (pour les réseaux avec ReLU ou Leaky ReLU) :

Avantages : L'initialisation de He est conçue pour améliorer la performance de ReLU, en ajustant l'échelle des poids en fonction du nombre de neurones.
Pratique générale :
Utiliser He initialization pour les réseaux utilisant ReLU.
Utiliser Xavier/Glorot pour les réseaux avec sigmoid ou tanh.


4. Dropout
Le dropout est une technique de régularisation qui consiste à désactiver aléatoirement une fraction des neurones pendant l'entraînement pour éviter le sur-apprentissage.

Bonnes pratiques :
Taux de dropout : Typiquement entre 0.2 et 0.5.
Un taux de 0.2 à 0.3 est recommandé dans les réseaux classiques.
0.5 peut être utilisé dans des réseaux très profonds ou pour des tâches de classification très complexes (ex. : vision par ordinateur).
Utilisation : Ajouter du dropout après les couches denses ou convolutives, particulièrement dans les couches profondes.
Pratique générale :
0.3 ou 0.5 dans les couches profondes pour éviter le sur-apprentissage, particulièrement pour les grands modèles ou ensembles de données complexes.


5. Nombre de couches cachées
Le nombre de couches cachées influence la capacité du réseau à apprendre des représentations complexes.

Bonnes pratiques :
Petits réseaux (problèmes simples) : Commencer avec 2 à 3 couches cachées.
Réseaux profonds (Deep Learning) : Pour des tâches comme la vision par ordinateur (CNN) ou le traitement du langage naturel (RNN), des réseaux avec 10 à 50 couches peuvent être utilisés.
Pratique générale :
Commencer avec 2-3 couches pour les réseaux simples.
Si nécessaire, augmenter progressivement jusqu’à 10-20 couches pour des tâches complexes.


6. Optimizer (Optimiseur)
L’optimiseur détermine comment les poids sont mis à jour en fonction des gradients calculés.

Bonnes pratiques :
Adam :

Avantages : Un des optimisateurs les plus populaires et les plus efficaces, ajuste dynamiquement le taux d’apprentissage pour chaque paramètre. Il est généralement utilisé pour de nombreux types de tâches.
Taux d’apprentissage recommandé : 0.001 pour commencer.
SGD avec momentum :

Avantages : Utilisé lorsque vous souhaitez un contrôle plus fin sur le taux d'apprentissage. Le momentum aide à accélérer la convergence en ajoutant une composante qui mémorise la direction précédente.
Taux d’apprentissage recommandé : 0.01 à 0.1.
RMSprop :

Avantages : Bon choix pour les problèmes séquentiels ou les architectures récurrentes (RNN).
Taux d’apprentissage recommandé : 0.001.
Pratique générale :
Adam avec un taux de 0.001 pour commencer, sauf si vous avez des raisons spécifiques d'utiliser un autre optimiseur.
Si vous utilisez SGD, envisagez d’ajouter momentum pour améliorer la convergence.

7. Learning rate (Taux d’apprentissage)
Le learning rate détermine la taille des étapes faites lors de la mise à jour des poids pendant l’entraînement.

Bonnes pratiques :
Taux d’apprentissage fixe : Le taux d’apprentissage doit être suffisamment bas pour que le modèle converge de manière stable mais pas trop faible pour éviter un long entraînement.
Valeur initiale : 0.001 pour la plupart des optimisateurs comme Adam.
Pour SGD, un taux de 0.01 à 0.1 est couramment utilisé.
Réduction du learning rate :
Utiliser des techniques comme la réduction du taux d’apprentissage lors de l’atteinte d’un plateau (ex. : ReduceLROnPlateau en Keras).
Pratique générale :
0.001 pour Adam et ajustez en fonction des performances.
0.01 à 0.1 pour SGD, surtout si vous utilisez des techniques comme le momentum.
