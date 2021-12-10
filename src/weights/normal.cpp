//
// Created by R on 11/14/2021.
//

#include "include/weights/normal.hpp"
#include <random>

namespace net
{
    namespace weights
    {
        void Normal::init_weights(Matrix &weights, double optional)
        {
            weights = Matrix::NullaryExpr(weights.rows(), weights.cols(), [&] {
                return Normal::random_number(0, 1);
            });
        }


        double Normal::random_number(double minimum, double maximum)
        {
            std::random_device                     random;
            std::mt19937                           mt(random());
            std::uniform_real_distribution<double> distribution(minimum, maximum);

            return distribution(mt);
        }
    }
}