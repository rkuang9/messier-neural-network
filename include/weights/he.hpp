//
// Created by R on 11/14/2021.
//

#ifndef NEURALNETWORK_HE_HPP
#define NEURALNETWORK_HE_HPP

#include <lib/Eigen/Dense>
#include "include/layer/layer.hpp"

namespace net
{
    namespace weights
    {
        class He
        {
            typedef Eigen::MatrixX<double> Matrix;
        public:
            static void init_weights(Matrix &weights, double input_units);
        };
    }
}


#endif //NEURALNETWORK_HE_HPP
