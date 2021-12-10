//
// Created by R on 12/6/2021.
//

#include "include/activation/softplus.hpp"

namespace net
{
    namespace activation
    {

        ColumnVector Softplus::activate(ColumnVector &z)
        {
            return (1 + z.array().exp()).log();
        }


        ColumnVector Softplus::gradient(ColumnVector &z)
        {
            return 1 / (1 + (-z.array()).exp()); // sigmoid function
        }
    }
}
