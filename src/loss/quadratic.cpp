//
// Created by R on 11/15/2021.
//

#include "include/loss/quadratic.hpp"

namespace net
{
    namespace loss
    {
        double Quadratic::loss(const ColumnVector &a, const ColumnVector &expected)
        {
            throw std::invalid_argument("Quadratic::loss() not implemented yet");
            return 0;
        }


        ColumnVector Quadratic::gradient(const ColumnVector &a, const ColumnVector &expected)
        {
            return (a - expected);
        }
    }
}


