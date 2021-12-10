//
// Created by R on 11/13/2021.
//

#ifndef NEURALNETWORK_DENSE_HPP
#define NEURALNETWORK_DENSE_HPP

#include "include/layer/layer.hpp"
#include "lib/Eigen/Dense"

namespace net
{
    template<typename Activation, typename Weights>
    class Dense : public Layer
    {
    public:
        explicit Dense(unsigned inputs, unsigned units);

        explicit Dense(unsigned inputs, unsigned units, bool use_bias);

        ~Dense() override = default;

        ColumnVector &neurons_activated() override;

        ColumnVector &neurons_output() override;

        ColumnVector &error() override;

        Matrix &get_weights() override;

    public: // TODO: public for testing, change back to private
        void feedforward(const ColumnVector &previous_a) override;

        void backpropagate(const ColumnVector &left_hadamard,
                           const ColumnVector &previous_activation) override;

        void update(optimizer::Optimizer &opt) override;

        void init_matrices(unsigned inputs, unsigned units);

        Matrix       weights;
        Matrix       nabla_weights; // nabla means gradient
        ColumnVector z;
        ColumnVector a;
        ColumnVector delta;

        ColumnVector bias;
        ColumnVector nabla_bias;
        bool         use_bias = false;

    };
}

#include "src/layer/dense.ipp"

#endif //NEURALNETWORK_DENSE_HPP
