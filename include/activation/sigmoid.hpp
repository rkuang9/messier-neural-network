//
// Created by R on 11/12/2021.
//

#ifndef NEURALNETWORK_SIGMOID_HPP
#define NEURALNETWORK_SIGMOID_HPP

#include "include/typedefs.hpp"

namespace net
{
    namespace activation
    {
        class Sigmoid
        {
        public:
            // z - neuron vector before activation
            // a - variable to assign activated neuron vector to
            static ColumnVector activate(ColumnVector &z);

            // z - neuron vector before activation
            // g - gradient vector
            static ColumnVector gradient(ColumnVector &z);
        };
    }
}

#endif //NEURALNETWORK_SIGMOID_HPP
