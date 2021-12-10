//
// Created by R on 11/12/2021.
//

#include "include/activation/sigmoid.hpp"

namespace net
{
    namespace activation
    {
        ColumnVector Sigmoid::activate(const ColumnVector &z)
        {
            // for each element via array() apply the sigmoid formula 1 / (1 + e^-z)
            return 1 / (1 + (-z.array()).exp());
        }


        ColumnVector Sigmoid::gradient(const ColumnVector &z)
        {
            return (1 / (1 + (-z.array()).exp())) * (1 - (1 / (1 + (-z.array()).exp())));
        }
    }
}