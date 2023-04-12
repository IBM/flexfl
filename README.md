## Federated Learning with Flexible Control (FlexFL)

This is the official code of the paper S. Wang, J. Perazzone, M. Ji, K. Chan, "[Federated learning with flexible control](https://arxiv.org/abs/2212.08496)," in IEEE International Conference on Computer Communications (INFOCOM), May 2023.
```
@inproceedings{wang2022federated,
  title={Federated Learning with Flexible Control},
  author={Wang, Shiqiang and Perazzone, Jake and Ji, Mingyue and Chan, Kevin S},
  booktitle={IEEE International Conference on Computer Communications (INFOCOM)},
  year={2022}
}
```


The code was run successfully in the following environment: Python 3.8, PyTorch 1.7, Torchvision 0.8.1

See `config.py` for all the configurations. Some examples are as follows.

FashionMNIST dataset with 20 random seeds (proposed method in the first line and baseline with k<sub>r</sub>=0.01 in the second line):
```
python3 main.py -data fashion -compression-adaptive-method lyapunov -seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 -out results_fashion_proposed

python3 main.py -data fashion -compression-adaptive-method fixed-0.01 -seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 -out results_fashion_baseline_0.01
```

CIFAR-10 dataset with 5 random seeds (proposed method in the first line and baseline with k<sub>r</sub>=0.01 in the second line):
```
python3 main.py -data cifar -compression-adaptive-method lyapunov -seeds 1,2,3,4,5 -out results_cifar10_proposed

python3 main.py -data cifar -compression-adaptive-method fixed-0.01 -seeds 1,2,3,4,5 -out results_cifar10_baseline_0.01
```

The results are saved in `results_*.csv` by default and the prefix `results` can be changed to another value by specifying the `-out` argument.

