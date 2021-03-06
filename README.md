[![GoDoc](https://godoc.org/github.com/golang/gddo?status.svg)](https://godoc.org/github.com/LdDl/cnns)
[![Build Status](https://travis-ci.com/LdDl/cnns.svg?branch=master)](https://travis-ci.com/LdDl/cnns)
[![Go Report Card](https://goreportcard.com/badge/github.com/LdDl/cnns)](https://goreportcard.com/report/github.com/LdDl/cnns)

![alt text](https://raw.githubusercontent.com/LdDl/cnns/master/cnns_png.png)

# CNNs #
CNNS (Convolutional Neural Networks) is a little package for developing simple neural networks, such as CNN (you don't say?) and MLP.

## It has been made only for studying purposes. Do not use it in production!
## Any PR's (new layers, learning optimizators, Excel-examples, bug-fixes) to improve this library will be appreciated.
**Currently I have not that much time to work on this (but this project _NOT_ abandoned)**

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Support](#support)

## Features

- CNN (convolutional neural network)
- MLP (multilayer perceptron)

## Installation

Installation is pretty simple:
```go
go get github.com/LdDl/cnns
```

## Usage

Just look into [examples folder](examples)

For some peoples it is realy hard to understand algorithms without step-by-step examples. So we provide some:

* Step-by-step feedforward and backpropagation calculations in CNN are made in file of Excel format (xlsx): [step_by_step_cnn(inertia).xlsx](examples/simple_cnn/step_by_step_cnn(inertia).xlsx). You can check associated example [examples/simple_cnn/main.go](examples/simple_cnn/main.go) also. Notice, that CNN in the example uses inertia extension. 

* Step-by-step feedforward and backpropagation calculations in MLP:
    1) Without inertia extension (not used in main code of repository):

        [step_by_step_mlp.xlsx](examples/simple_mlp/step_by_step_mlp.xlsx)
    2) With inertia extension (this one is used currently):

        [step_by_step_mlp(inertia).xlsx](examples/simple_mlp/step_by_step_mlp(inertia).xlsx)
        and associated example [examples/simple_mlp/main.go](examples/simple_mlp/main.go)

There are 3 epochs provided (see tabs: Step_1, Step_2, Step_3)) for each net also.

## Support

If you have troubles or questions please [open an issue](https://github.com/LdDl/cnns/issues/new).
