//
// Created by Macross on 12/4/2021.
//

#include "include/optimizer/adam.hpp"

namespace net
{
    namespace optimizer
    {
        Adam::Adam(double learning_rate, unsigned int batch_size)
                : batch_size(batch_size)
        {
        }


        Adam::Adam(double learning_rate, unsigned int batch_size, double beta_decay_m, double beta_decay_v)
                : batch_size(batch_size),
                  beta_decay_m(beta_decay_m),
                  beta_decay_v(beta_decay_v)
        {
        }


        void Adam::update(Matrix &weights, Matrix &nabla_weights)
        {

        }
    }
}