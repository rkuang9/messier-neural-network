//
// Created by R on 12/8/2021.
//

#include "include/loss/cross_entropy.hpp"

namespace net
{
    namespace loss
    {
        double CrossEntropy::loss(const ColumnVector &a, const ColumnVector &expected)
        {
            // if "z" has 0 values, an error may error due to natural log of zero
            return (
                expected.array() * a.array().log() +
                (1 - expected.array()) * (1 - a.array()).log()
            ).sum();
        }


        ColumnVector CrossEntropy::gradient(const ColumnVector &a, const ColumnVector &expected)
        {
            // add 1e-4 to the denominator to prevent division by zero
            // be wary of using this loss function since it can return
            // massively negative values
            return (a.array() - expected.array()) / ((1 - a.array()) * (a.array() + 1e-4));
        }
    }
}
