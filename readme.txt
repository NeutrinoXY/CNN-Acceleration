L'exécution du script main.py nécessite les libraires Numpy et PIL. Ce script utilise les fichiers layers.py et weights.py, qui contiennent respectivement le code des objets correspondant aux couches du réseau, et les poids des différents neurones des couches de convolution et du perceptron. Le réseau de neurones créé dans le script main.py correspond à celui proposé à la fin du sujet.
Le réseau parvient à détecter les classes associées aux images avec un taux de succès situé entre 70 et 80%.

Afin que le script trouve les images à analyser, il faut extraire le fichier contenant les images cifar10 (version Python) à la racine du dossier. Le fichier Python peut être trouvé sur la page : https://www.cs.toronto.edu/~kriz/cifar.html

Pour exécuter le script, il suffit de taper la ligne de commande :

python main.py

Puis d'entrer le nombre d'images issues de CIFAR10 à analyser (entre 1 et 10000).

Résultats pour l'analyse de 200 images :
Taux de succès : 73,6%
