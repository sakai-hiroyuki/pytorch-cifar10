# pytorch-cifar10
[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Optimizers
|Optimizer|Implementation|
|----|:----:|
|SGD|https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD|
|Momentum|https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD|
|RMSprop|https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop|
|[Adam](https://arxiv.org/abs/1412.6980)|https://pytorch.org/docs/stable/generated/torch.optim.Adam.html|
|[AMSGrad](https://arxiv.org/abs/1904.09237)|https://pytorch.org/docs/stable/generated/torch.optim.Adam.html|
|[AdaBelief](https://arxiv.org/abs/2010.07468)|https://github.com/juntang-zhuang/Adabelief-Optimizer|

## References
```bibtex
@book{
  title={現場で使える！PyTorch開発入門 深層学習モデルの作成とアプリケーションへの実装},
  author={杜世橋},
  year={2018},
  publisher={翔泳社}
}

@inproceedings{
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}

@inproceedings{
  title={Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift},
  author={Ioffe, Sergey and Szegedy, Christian},
  booktitle={International Conference on Machine Learning},
  pages={448--456},
  year={2015},
}

@inproceedings{
  title={Don't decay the learning rate, increase the batch size},
  author={Smith, Samuel L and Kindermans, Pieter-Jan and Ying, Chris and Le, Quoc V},
  journal={International Conference on Learning Representations},
  year={2018}
}

@article{
  title={Improved regularization of convolutional neural networks with cutout},
  author={DeVries, Terrance and Taylor, Graham W},
  journal={arXiv preprint arXiv:1708.04552},
  year={2017}
}
```
