//
// Created by R on 12/6/2021.
//

#include "include/activation/softsign.hpp"

namespace net
{
    namespace activation
    {

        ColumnVector Softsign::activate(const ColumnVector &z)
        {
            return z.array() / (1 + z.array().abs());
        }


        ColumnVector Softsign::gradient(const ColumnVector &z)
        {
            return 1 / ((1 + z.array().abs()).pow(2));
        }
    }
}