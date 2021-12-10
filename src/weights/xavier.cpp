//
// Created by R on 11/14/2021.
//

#include "include/weights/xavier.hpp"
#include "include/weights/normal.hpp"
#include <cmath>

namespace net
{
    void Xavier::init_weights(Matrix &weights, double input_units)
    {
        Normal::init_weights(weights, -1);
        weights.noalias() = weights * sqrt(1 / input_units);
    }
}

