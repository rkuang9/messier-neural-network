//
// Created by R on 12/6/2021.
//

#ifndef NEURALNETWORK_SOFTPLUS_HPP
#define NEURALNETWORK_SOFTPLUS_HPP

#include "include/typedefs.hpp"

namespace net
{
    namespace activation
    {
        class Softplus
        {
        public:
            // z - neuron vector before activation
            // a - variable to assign activated neuron vector to
            static ColumnVector activate(ColumnVector &z);

            static ColumnVector gradient(ColumnVector &z);
        };
    }
}


#endif //NEURALNETWORK_SOFTPLUS_HPP
