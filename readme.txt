L'exécution du script main.py nécessite les libraires Numpy et PIL. Ce script utilise les fichiers layers.py et weights.py, qui contiennent respectivement le code des objets correspondant aux couches du réseau, et les poids des différents neurones des couches de convolution et du perceptron. Le réseau de neurones créé dans le script main.py correspond à celui proposé à la fin du sujet.
Malheureusement, le réseau ne parvient pas à détecter la classe des images provenant de la base de données Cifar-10. Chaque couche fonctionnement individuellement, une hypothèse pouvant expliquer l'échec du réseau serait un mauvais reshape du layer FullyConnected, qui doit être conforme aux poids fournis pour le perceptron.

Résultats pour l'analyse de 200 images :
30 classes détectées avec succès.
Taux de succès : 15%.
