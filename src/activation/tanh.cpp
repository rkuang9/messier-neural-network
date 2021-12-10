//
// Created by R on 11/13/2021.
//

#include "include/activation/tanh.hpp"


namespace net
{
    namespace activation
    {
        ColumnVector Tanh::activate(ColumnVector &z)
        {
            return z.array().tanh();
        }


        ColumnVector Tanh::gradient(ColumnVector &z)
        {
            return (1 - (z.array().tanh()).pow(2));
        }
    }
}