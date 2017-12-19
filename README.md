# Learning-with-Feature-Fide-Information
Travail de Bachelor

Joao Antonio Candido Ramos


Le but de mon travail était d'implémenter l'approche analytique, proposée 
par Amina Mollaysa ([papier](https://arxiv.org/abs/1703.02570)),
avec deux librairies python : [TensorFlow](https://www.tensorflow.org/) et [Pytorch](http://pytorch.org/). 

## Prérequis
Nous avons utilisé python 3.5 et les librairies suivantes : 

 - Numpy
 - h5py
 - scipy
 - Matplotlib
 - TensorFlow
 - Pytorch
 
La façon la plus simple d'installer ces librairies est sans doute grâce à [Anaconda](https://www.anaconda.com/download/) et à la commande :

```
$ conda install nom_librairie
```

## TensorFlow

Pour obtenir la liste des paramètres et des informations les concernant :
```
python tf.py -h
```

Un exemple de commande pour entraîner son réseau de neuronnes pendant 4000 époques : 
```
python tf.py filename.mat -e 4000
```

Le dossier logs contiendra les logs pour une visualisation sur Tensorboard. 
Pour visualiser il faut executer, dans le répertoire courant à tf.py, la commande suivante :
```
tensorboard --logdir=./logs
```
et se rendre ensuite sur le [port 6006 local](http://localhost:6006/).

Le dossier checkpoints contiendra la sauvegarde de l'etat du réseau lorsque le meilleur taux de reussite 
dans l'ensemble de validation a été trouvé.


## Pytorch
Pour obtenir la liste des paramètres et des informations les concernant :
```
python pytorch.py -h
```
Un exemple de commande pour entraîner son réseau de neuronnes pendant 4000 époques : 
```
python pytorch.py filename.mat -e 4000
```

Le dossier logs contiendra des graphiques générées grâce à matplotlib, sur le coût, taux de reussite 
de l'ensemble d'entrainement et le taux de reussite de l'ensemble de validation. 

Le dossier checkpoints contiendra la sauvegarde de l'etat du réseau lorsque le meilleur taux de reussite 
dans l'ensemble de validation a été trouvé.
