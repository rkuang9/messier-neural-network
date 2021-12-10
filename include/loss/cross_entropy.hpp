//
// Created by R on 12/8/2021.
//

#ifndef NEURALNETWORK_CROSS_ENTROPY_HPP
#define NEURALNETWORK_CROSS_ENTROPY_HPP

#include "include/typedefs.hpp"

namespace net
{
    namespace loss
    {
        class CrossEntropy
        {
        public:
            static double loss(const ColumnVector &a, const ColumnVector &expected);

            static ColumnVector gradient(const ColumnVector &a, const ColumnVector &expected);
        };
    }
}

#endif //NEURALNETWORK_CROSS_ENTROPY_HPP
