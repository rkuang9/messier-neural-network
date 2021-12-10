//
// Created by R on 11/15/2021.
//

#include "include/model.hpp"

namespace net
{
    template<typename Loss>
    Model<Loss>::Model(std::initializer_list<Layer *> layers) : layers(layers)
    {
    }


    template<typename Loss>
    Model<Loss>::~Model()
    {
        for (Layer *i: this->layers) {
            delete i;
        }
    }


    template<typename Loss>
    void Model<Loss>::add_layer(Layer *new_layer)
    {
        this->layers.push_back(new_layer);
    }


    template<typename Loss>
    void Model<Loss>::train(optimizer::Optimizer &opt,
                            unsigned epochs,
                            std::vector<std::vector<double>> training_data,
                            std::vector<std::vector<double>> expected_output)
    {
        if (training_data.size() != expected_output.size()) {
            throw std::invalid_argument("(model::train) not 1:1 training_data to expected_output");
        }

        if (training_data.front().size() != this->layers.front()->neurons_activated().rows()) {
            throw std::invalid_argument(
                "(model::train) training sample size " +
                std::to_string(training_data.front().size()) +
                " != input layer size " +
                std::to_string(this->layers.front()->neurons_activated().rows()));
        }

        for (unsigned age = 0; age < epochs; ++age) {
            for (unsigned i = 0; i < training_data.size(); ++i) {
                ColumnVector data_set = Eigen::Map<ColumnVector>(training_data[i].data(), training_data[i].size());
                this->feedforward(data_set);

                ColumnVector expect = Eigen::Map<ColumnVector>(expected_output[i].data(), expected_output[i].size());
                this->backpropagate(expect);

                this->update(opt);
            }
        }
    }


    template<typename Loss>
    void Model<Loss>::train(optimizer::Optimizer &opt,
                            unsigned epochs,
                            std::vector<ColumnVector> &training_data,
                            std::vector<ColumnVector> &expected_output)
    {
        if (training_data.size() != expected_output.size()) {
            throw std::invalid_argument("(model::train) not 1:1 training_data to expected_output");
        }

        if (training_data.front().rows() != this->layers.front()->neurons_activated().rows()) {
            throw std::invalid_argument(
                "(model::train) training sample size " +
                std::to_string(training_data.front().rows()) +
                " != input layer size " +
                std::to_string(this->layers.front()->neurons_activated().rows()));
        }

        for (unsigned age = 0; age < epochs; ++age) {
            for (unsigned i = 0; i < training_data.size(); ++i) {
                this->feedforward(training_data[i]);
                this->backpropagate(expected_output[i]);

                this->update(opt);
            }
        }
    }


    template<typename Loss>
    void Model<Loss>::feedforward(ColumnVector &input)
    {
        // sets input layer "z" and "a" column vectors
        this->layers.front()->feedforward(input);

        for (unsigned i = 1; i < this->layers.size(); ++i) {
            // use previous layer's activation "a" as inputs "z" for layer i
            this->layers[i]->feedforward(this->layers[i - 1]->neurons_activated());
        }
    }


    template<typename Loss>
    void Model<Loss>::backpropagate(ColumnVector &expected_output)
    {
        // on output layer first
        this->layers.back()->backpropagate(
            Loss::gradient(this->layers.back()->neurons_activated(), expected_output),
            this->layers[this->layers.size() - 2]->neurons_activated()
        );

        // backpropagate, starting from last to first hidden layer
        for (unsigned i = this->layers.size() - 2; i != 0; --i) {
            this->layers[i]->backpropagate(
                // next layer's weights transposed * next layer's delta (error)
                this->layers[i + 1]->get_weights().transpose() * this->layers[i + 1]->error(),
                // previous layer's activations, for calculating nabla weights of layer i
                this->layers[i - 1]->neurons_activated()
            );
        }

        // input layer has no weights and does not need backpropagating
    }


    template<typename Loss>
    void Model<Loss>::update(optimizer::Optimizer &opt)
    {
        for (unsigned i = 1; i < this->layers.size(); ++i) {
            this->layers[i]->update(opt);
        }
    }


    template<typename Loss>
    ColumnVector Model<Loss>::output()
    {
        return this->layers.back()->neurons_output();
    }


    template<typename Loss>
    ColumnVector Model<Loss>::predict(ColumnVector &input)
    {
        this->feedforward(input);
        return this->layers.back()->neurons_output();
    }


    template<typename Loss>
    ColumnVector Model<Loss>::normalize(ColumnVector &vector, double data_max_val, double data_min_val,
                                        double normalized_max, double normalized_min)
    {
        return (((vector.array() - data_min_val) / (data_max_val - data_min_val)) *
                (normalized_max - normalized_min)) + normalized_min;
    }
}
