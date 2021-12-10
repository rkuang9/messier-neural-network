//
// Created by R on 11/15/2021.
//

#ifndef NEURALNETWORK_QUADRATIC_HPP
#define NEURALNETWORK_QUADRATIC_HPP

#include "include/typedefs.hpp"

namespace net
{
    namespace loss
    {
        class Quadratic
        {
        public:
            static double loss(const ColumnVector &a, const ColumnVector &expected);

            static ColumnVector gradient(const ColumnVector &a, const ColumnVector &expected);
        };
    }
}


#endif //NEURALNETWORK_QUADRATIC_HPP
