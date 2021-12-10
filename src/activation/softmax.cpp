//
// Created by R on 12/5/2021.
//

#include "include/activation/softmax.hpp"
#include <iostream>
namespace net
{
    namespace activation
    {
        ColumnVector Softmax::activate(ColumnVector &z)
        {
            return z.array().exp() / z.array().exp().sum();
        }


        ColumnVector Softmax::gradient(ColumnVector &z)
        {
            // same as sigmoid derivative where it's dSoftmax(z)/d(z) = Softmax(z) * (1 - Softmax(z))
            // derivation may seem daunting, but it's just the quotient rule
            // and substituting the softmax back in where possible
            return (z.array().exp() / z.array().exp().sum()) * (1 - (z.array().exp() / z.array().exp().sum()));
        }
    }
}