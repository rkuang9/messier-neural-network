//
// Created by R on 11/14/2021.
//

#include "include/weights/he.hpp"
#include "include/weights/normal.hpp"
#include <cmath>
#include <iostream>

namespace net
{
    namespace weights
    {
        void He::init_weights(Matrix &weights, double input_units)
        {
            Normal::init_weights(weights, -1);
            weights.noalias() = weights * sqrt(2 / input_units);
        }
    }
}