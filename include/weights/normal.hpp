//
// Created by R on 11/14/2021.
//

#ifndef NEURALNETWORK_NORMAL_HPP
#define NEURALNETWORK_NORMAL_HPP

#include <lib/Eigen/Dense>
#include "include/layer/layer.hpp"

namespace net
{
    class Normal
    {
        typedef Eigen::MatrixX<double> Matrix;
    public:
        static void init_weights(Matrix &weights, double optional);

        static double random_number(double minimum, double maximum);
    };
}


#endif //NEURALNETWORK_NORMAL_HPP
