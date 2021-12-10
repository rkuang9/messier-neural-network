//
// Created by R on 11/12/2021.
//

#include "include/activation/relu.hpp"

namespace net
{
    namespace activation
    {
        ColumnVector ReLU::activate(ColumnVector &z)
        {
            return z.array().max(0);
        }


        ColumnVector ReLU::gradient(ColumnVector &z)
        {
            ColumnVector gradient;
            gradient.resize(z.size());

            for (unsigned i = 0; i < z.size(); ++i) {
                gradient[i] = (z[i] >= 0) ? 1 : 0;
            }

            return gradient;
        }
    }
}