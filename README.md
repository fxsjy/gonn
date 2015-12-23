GoNN [![GoDoc](https://godoc.org/github.com/fxsjy/gonn/gonn?status.svg)](https://godoc.org/github.com/fxsjy/gonn/gonn)
========
Neural Network in GoLang

Feature
=======
* BackPropagation Network / RBF Network / Perceptron Network
* Parallel BackPropagation Network (each neural has its own go-routine)

Benchmark
=======
* Dataset: MNIST Acurrency Rate : 98.2% (800 hidden nodes)
* Actually, you can get 96.9% using 100 hidden nodes in just three minutes of trainning


TODO
=======
* currently, the parallel version is much slower than the tranditional one, maybe caused by the cost of context switch of threads

