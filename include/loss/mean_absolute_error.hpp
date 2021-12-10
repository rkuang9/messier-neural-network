//
// Created by R on 12/8/2021.
//

#ifndef NEURALNETWORK_MEAN_ABSOLUTE_ERROR_HPP
#define NEURALNETWORK_MEAN_ABSOLUTE_ERROR_HPP

#include "include/typedefs.hpp"

namespace net
{
    namespace loss
    {
        class MeanAbsoluteError
        {
        public:
            static double loss(const ColumnVector &a, const ColumnVector &expected);

            static ColumnVector gradient(const ColumnVector &a, const ColumnVector &expected);
        };
    }
}


#endif //NEURALNETWORK_MEAN_ABSOLUTE_ERROR_HPP
