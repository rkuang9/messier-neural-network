//
// Created by R on 11/13/2021.
//

#ifndef NEURALNETWORK_TANH_HPP
#define NEURALNETWORK_TANH_HPP

#include "include/typedefs.hpp"

namespace net
{
    namespace activation
    {

        class Tanh
        {
        public:
            // z - neuron vector before activation
            // a - variable to assign activated neuron vector to
            static ColumnVector activate(ColumnVector &z);

            static ColumnVector gradient(ColumnVector &z);
        };
    }
}


#endif //NEURALNETWORK_TANH_HPP
