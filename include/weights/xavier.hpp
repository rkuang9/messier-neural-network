//
// Created by R on 11/14/2021.
//

#ifndef NEURALNETWORK_XAVIER_HPP
#define NEURALNETWORK_XAVIER_HPP

#include <lib/Eigen/Dense>
#include "include/layer/dense.hpp"

namespace net
{
    namespace weights
    {
        class Xavier
        {
            typedef Eigen::MatrixX<double> Matrix;
        public:
            static void init_weights(Matrix &weights, double input_units);
        };
    }
}


#endif //NEURALNETWORK_XAVIER_HPP
