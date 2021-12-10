//
// Created by Macross on 12/4/2021.
//

#ifndef NEURALNETWORK_ADAM_HPP
#define NEURALNETWORK_ADAM_HPP

#include "include/typedefs.hpp"
#include "include/optimizer/optimizer.hpp"

namespace net
{
    namespace optimizer
    {
        class Adam : private Optimizer
        {
        public:
            Adam(double learning_rate, unsigned batch_size);

            Adam(double learning_rate, unsigned batch_size, double beta_decay1, double beta_decay2);

            void update(Matrix &weights, const Matrix &nabla_weights) override;

            void update(ColumnVector &bias, const ColumnVector &nabla_bias) override;

        private:
            unsigned batch_size;
            double beta_decay_m = 0.9;
            double beta_decay_v = 0.999;
        };
    }
}


#endif //NEURALNETWORK_ADAM_HPP
