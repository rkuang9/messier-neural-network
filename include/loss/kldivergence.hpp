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
            static double loss(const ColumnVector &a, const ColumnVector &expected);

            static ColumnVector gradient(const ColumnVector &a, const ColumnVector &expected);
        };

    }
}


#endif //NEURALNETWORK_KLDIVERGENCE_HPP
