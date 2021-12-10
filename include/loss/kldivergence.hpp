//
// Created by R on 12/8/2021.
//

#ifndef NEURALNETWORK_KLDIVERGENCE_HPP
#define NEURALNETWORK_KLDIVERGENCE_HPP

#include "include/typedefs.hpp"

namespace net
{
    namespace loss
    {
        class KLDivergence
        {
        public:
            static double loss(ColumnVector &z, ColumnVector &expected);

            static ColumnVector gradient(ColumnVector &a, ColumnVector &expected);
        };

    }
}


#endif //NEURALNETWORK_KLDIVERGENCE_HPP
