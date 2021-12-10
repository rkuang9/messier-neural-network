//
// Created by R on 11/13/2021.
//

#ifndef NEURALNETWORK_COMMON_HPP
#define NEURALNETWORK_COMMON_HPP

#ifdef NN_DEBUG
#include <iostream>
#endif

#include "model.hpp"

// activation functions
#include "include/activation/sigmoid.hpp"
#include "include/activation/tanh.hpp"
#include "include/activation/relu.hpp"
#include "include/activation/leakyrelu.hpp"
#include "include/activation/softmax.hpp"
#include "include/activation/swish.hpp"
#include "include/activation/softplus.hpp"
#include "include/activation/softsign.hpp"

// weight initializations
#include "include/weights/normal.hpp"
#include "include/weights/xavier.hpp"
#include "include/weights/he.hpp"

// layer types
#include "include/layer/layer.hpp" // base class
#include "include/layer/dense.hpp"
#include "include/layer/input.hpp"

// cost functions
#include "include/loss/quadratic.hpp"
#include "include/loss/cross_entropy.hpp"
#include "include/loss/kldivergence.hpp"
#include "include/loss/mean_absolute_error.hpp"

// optimizers
#include "include/optimizer/optimizer.hpp" // base class
#include "include/optimizer/sgd.hpp"
#include "include/optimizer/adam.hpp"

#endif //NEURALNETWORK_COMMON_HPP
