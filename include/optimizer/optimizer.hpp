//
// Created by R on 11/17/2021.
//

#ifndef NEURALNETWORK_OPTIMIZER_HPP
#define NEURALNETWORK_OPTIMIZER_HPP

#include "include/typedefs.hpp"

namespace net
{
    namespace optimizer
    {
        class Optimizer
        {
        public:
            virtual void update(Matrix &weights, Matrix &nabla_weights) = 0;

            virtual void update(ColumnVector &bias, ColumnVector &nabla_bias) = 0;

            double learning_rate = 0.01;
            double epsilon_smoothing = 1e-8;
        };
    }
}

#endif //NEURALNETWORK_OPTIMIZER_HPP
