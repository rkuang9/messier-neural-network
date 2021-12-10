//
// Created by R on 11/17/2021.
//

#include "include/optimizer/sgd.hpp"

namespace net
{
    namespace optimizer
    {
        SGD::SGD(double learning_rate) : learning_rate(learning_rate)
        {
            // nothing to do
        }


        void SGD::update(Matrix &weights, const Matrix &nabla_weights)
        {
            weights -= (this->learning_rate * nabla_weights);
        }


        void SGD::update(ColumnVector &bias, const ColumnVector &nabla_bias)
        {
            bias -= (this->learning_rate * nabla_bias);
        }
    }
}