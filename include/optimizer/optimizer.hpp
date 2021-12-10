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
            virtual void update(Matrix &weights, const Matrix &nabla_weights) = 0;

            virtual void update(ColumnVector &bias, const ColumnVector &nabla_bias) = 0;
        };
    }
}

#endif //NEURALNETWORK_OPTIMIZER_HPP
