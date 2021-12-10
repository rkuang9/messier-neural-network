//
// Created by R on 11/12/2021.
//

#include "include/activation/leakyrelu.hpp"

namespace net
{
    namespace activation
    {
        ColumnVector LeakyReLU::activate(ColumnVector &z)
        {
            ColumnVector relu;
            relu.resize(z.size());

            for (unsigned i = 0; i < z.size(); ++i) {
                relu[i] = (z[i] > 0) ? 1 : (0.01 * z[i]);
            }

            return relu;
        }


        ColumnVector LeakyReLU::gradient(ColumnVector &z)
        {
            ColumnVector gradient;
            gradient.resize(z.size());

            for (unsigned i = 0; i < z.size(); ++i) {
                gradient[i] = (z[i] > 0) ? 1 : 0.01;
            }

            return gradient;
        }
    }
}