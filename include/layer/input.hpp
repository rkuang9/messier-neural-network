//
// Created by R on 11/13/2021.
//

#ifndef NEURALNETWORK_INPUT_HPP
#define NEURALNETWORK_INPUT_HPP

#include "include/layer/layer.hpp"
#include "lib/Eigen/Dense"

namespace net
{
    template<typename Activation>
    class Input : public Layer
    {
    public:
        explicit Input(unsigned units);

        ~Input() override = default;

        const ColumnVector &neurons_activated() const override;

        const ColumnVector &neurons_output() const override;

        ColumnVector &error() override {};

        Matrix &get_weights() override {};

    public:
        void feedforward(const ColumnVector &input_data) override;

        void backpropagate(const ColumnVector &previous_a,
                           const ColumnVector &next_delta) override {};

        void update(optimizer::Optimizer &opt) override {};

        ColumnVector z;
        ColumnVector a;
    };


}

#include "src/layer/input.ipp"

#endif //NEURALNETWORK_INPUT_HPP
