//
// Created by R on 11/13/2021.
//

#ifndef NEURALNETWORK_LEAKYRELU_HPP
#define NEURALNETWORK_LEAKYRELU_HPP

#include "include/typedefs.hpp"

namespace net
{
    namespace activation
    {
        class LeakyReLU
        {
        public:
            // z - neuron vector before activation
            // a - variable to assign activated neuron vector to
            static ColumnVector activate(const ColumnVector &z);

            static ColumnVector gradient(const ColumnVector &z);
        };
    }
}

#endif //NEURALNETWORK_LEAKYRELU_HPP
