//
// Created by R on 11/13/2021.
//

#include "include/layer/input.hpp"

namespace net
{
    template<typename Activation>
    Input<Activation>::Input(unsigned units)
    {
        this->z.resize(units);
        this->a.resize(units);
    }


    template<typename Activation>
    const ColumnVector &Input<Activation>::neurons_activated() const
    {
        return this->a;
    }


    template<typename Activation>
    const ColumnVector &Input<Activation>::neurons_output() const
    {
        return this->z;
    }


    template<typename Activation>
    void Input<Activation>::feedforward(const ColumnVector &input_data)
    {
        this->z = input_data;
        this->a = Activation::activate(this->z);
    }
}