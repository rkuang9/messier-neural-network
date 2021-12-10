//
// Created by R on 12/8/2021.
//

#include "include/loss/kldivergence.hpp"

namespace net
{
    namespace loss
    {
        double KLDivergence::loss(const ColumnVector &a, const ColumnVector &expected)
        {
            return (expected.array() * (expected.array() / a.array()).log10()).sum();
        }


        ColumnVector KLDivergence::gradient(const ColumnVector &a, const ColumnVector &expected)
        {
            // "a" must not have any zeroes
            return - (expected.array() / a.array());
        }
    }
}