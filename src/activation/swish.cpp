//
// Created by R on 12/6/2021.
//

#include "include/activation/swish.hpp"

namespace net
{
    namespace activation
    {
        ColumnVector Swish::activate(ColumnVector &z)
        {
            return z.array() * (1 / (1 + (-z.array()).exp()));
        }


        ColumnVector Swish::gradient(ColumnVector &z)
        {
            // swish + sigmoid * (1 - swish)
            return z.array() * (1 / (1 + (-z.array()).exp())) + // swish
                   (1 / (1 + (-z.array()).exp())) * // sigmoid
                   (1 - (z.array() * (1 / (1 + (-z.array()).exp())))); // 1 - swish
        }
    }
}