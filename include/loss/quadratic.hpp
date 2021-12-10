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
            static double loss(ColumnVector &z, ColumnVector &expected);

            static ColumnVector gradient(ColumnVector &a, ColumnVector &expected);
        };
    }
}


#endif //NEURALNETWORK_QUADRATIC_HPP
