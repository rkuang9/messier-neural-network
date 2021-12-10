//
// Created by R on 12/5/2021.
//

#ifndef NEURALNETWORK_SOFTMAX_HPP
#define NEURALNETWORK_SOFTMAX_HPP

#include "include/typedefs.hpp"

namespace net
{
    namespace activation
    {
        class Softmax
        {
        public:
            // z - neuron vector before activation
            // a - variable to assign activated neuron vector to
            static ColumnVector activate(ColumnVector &z);

            static ColumnVector gradient(ColumnVector &z);
        };
    }
}


#endif //NEURALNETWORK_SOFTMAX_HPP
