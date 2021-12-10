//
// Created by R on 11/13/2021.
//

#ifndef NEURALNETWORK_LAYER_HPP
#define NEURALNETWORK_LAYER_HPP

#include "include/typedefs.hpp"
#include "include/optimizer/optimizer.hpp"

namespace net
{
    // abstract class
    class Layer
    {
    public:
        typedef Eigen::VectorX<double> ColumnVector;
        typedef Eigen::MatrixX<double> Matrix;

        Layer() = default;

        virtual ~Layer() = default;

        virtual ColumnVector &neurons_activated() = 0;

        virtual ColumnVector &neurons_output() = 0;

        virtual ColumnVector &error() = 0;

        virtual void update(optimizer::Optimizer &opt) = 0;

        virtual Matrix &get_weights() = 0;

    public:
        virtual void feedforward(const ColumnVector &previous_a) = 0;

        virtual void backpropagate(const ColumnVector &vector,
                                   const ColumnVector &matrix) = 0;
    };

}

#endif //NEURALNETWORK_LAYER_HPP
