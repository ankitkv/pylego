# pylego

`pylego` lets you write readers and models that can be connected to different machine learning tasks. `pylego` can also handle your main training loop, including reporting results, dumping them for viewing in `tensorboard`, saving models regularly, loading saved models for visualization or resuming training, and so on.

The goal is to give you as much of a headstart with your machine learning project as possible, letting you then write only project-specific code in a way that you can try new models with varying objectives quickly and reliably.

**NOTE**: This project is still in active development, so there may be bugs to report! Tests and better documentation will also come soon.

## Dev installation
```
git clone https://github.com/ankitkv/pylego.git
cd pylego
pip install --user -e .
```

## Example projects

An MNIST classification project using `pylego` is included in `examples/mnist_classification`.

For a more advanced example, see [this implementation of TD-VAE][1].

[1]: https://github.com/ankitkv/TD-VAE
