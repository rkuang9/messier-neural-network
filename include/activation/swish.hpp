//
// Created by R on 12/6/2021.
//

#ifndef NEURALNETWORK_SWISH_HPP
#define NEURALNETWORK_SWISH_HPP

#include "include/typedefs.hpp"

namespace net
{
    namespace activation
    {
        class Swish
        {
        public:
            // z - neuron vector before activation
            // a - variable to assign activated neuron vector to
            static ColumnVector activate(const ColumnVector &z);

            static ColumnVector gradient(const ColumnVector &z);
        };
    }
}

#endif //NEURALNETWORK_SWISH_HPP
