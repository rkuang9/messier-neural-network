//
// Created by R on 11/15/2021.
//

#ifndef NEURALNETWORK_MODEL_HPP
#define NEURALNETWORK_MODEL_HPP

#include "include/layer/layer.hpp"
#include <vector>
#include "include/optimizer/optimizer.hpp"

namespace net
{
    template<typename Cost>
    class Model
    {
    public:
        Model(std::initializer_list<Layer *> layers);

        ~Model();

        void add_layer(Layer *new_layer);

        void train(optimizer::Optimizer &opt,
                   unsigned epochs,
                   std::vector<std::vector<double>> &training_data,
                   std::vector<std::vector<double>> &expected_output);

        void train(
            optimizer::Optimizer &opt,
            unsigned epochs,
            std::vector<ColumnVector> &training_data,
            std::vector<ColumnVector> &expected_output);


        void feedforward(const ColumnVector &input);

        void backpropagate(const ColumnVector &expected_output);

        void update(optimizer::Optimizer &opt);

        ColumnVector output();

        ColumnVector predict(ColumnVector &input);

        ColumnVector normalize(ColumnVector &vector,
                               double data_max_val, double data_min_val,
                               double normalized_max, double normalized_min);

    public:
        std::vector<Layer *> layers;
    };
}

#include "src/model.ipp"

#endif //NEURALNETWORK_MODEL_HPP
