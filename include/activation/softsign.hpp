//
// Created by R on 12/6/2021.
//

#ifndef NEURALNETWORK_SOFTSIGN_HPP
#define NEURALNETWORK_SOFTSIGN_HPP

#include "include/typedefs.hpp"

namespace net
{
    namespace activation
    {
        class Softsign
        {
        public:
            // z - neuron vector before activation
            // a - variable to assign activated neuron vector to
            static ColumnVector activate(ColumnVector &z);

            static ColumnVector gradient(ColumnVector &z);
        };
    }
}


#endif //NEURALNETWORK_SOFTSIGN_HPP
