//
// Created by R on 11/13/2021.
//

#include "include/layer/dense.hpp"

#ifdef NN_DEBUG

#include <iostream>

#endif

namespace net
{
	template<typename Activation, typename Weights>
	Dense<Activation, Weights>::Dense(unsigned inputs, unsigned units)
	{
		this->init_matrices(inputs, units);
	}


	template<typename Activation, typename Weights>
	Dense<Activation, Weights>::Dense(unsigned inputs, unsigned units, bool use_bias)
			:use_bias(use_bias)
	{
		this->init_matrices(inputs, units);

		if (use_bias) {
			// a column vector of size "units", each of value 0.01
			this->bias = ColumnVector::NullaryExpr(units, 1, [&] {
				return 0.01;
			});
		}
	}


	template<typename Activation, typename Weights>
    const ColumnVector &Dense<Activation, Weights>::neurons_activated() const
    {
		return this->a;
	}


	template<typename Activation, typename Weights>
    const ColumnVector &Dense<Activation, Weights>::neurons_output() const
    {
		return this->z;
	}


	template<typename Activation, typename Weights>
	ColumnVector &Dense<Activation, Weights>::error()
	{
		return this->delta;
	}


	template<typename Activation, typename Weights>
	Matrix &Dense<Activation, Weights>::get_weights()
	{
		return this->weights;
	}


	template<typename Activation, typename Weights>
	void Dense<Activation, Weights>::feedforward(const ColumnVector &previous_a)
	{
		this->z.noalias() = this->weights * previous_a;

		if (this->use_bias) {
			this->z.noalias() = this->z + this->bias;
		}

		this->a.noalias() = Activation::activate(this->z);
	}


	template<typename Activation, typename Weights>
	void Dense<Activation, Weights>::backpropagate(const ColumnVector &left_hadamard,
	                                               const ColumnVector &previous_activation)
	{
		this->delta.noalias() = left_hadamard.cwiseProduct(Activation::gradient(this->z));

		this->nabla_weights.noalias() = this->delta * previous_activation.transpose();

		if (this->use_bias) {
			this->nabla_bias = this->delta;
		}
	}


	template<typename Activation, typename Weights>
	void Dense<Activation, Weights>::update(optimizer::Optimizer &opt)
	{
		//this->weights -= (learning_rate * this->nabla_weights);
		opt.update(this->weights, this->nabla_weights);

		if (this->use_bias) {
			this->bias.noalias() -= 0.01 * this->nabla_bias;
			opt.update(this->bias, this->nabla_bias);
		}
	}


	template<typename Activation, typename Weights>
	void Dense<Activation, Weights>::init_matrices(unsigned inputs, unsigned units)
	{
		this->z.resize(units);
		this->a.resize(units);
		this->weights.resize(units, inputs);

		Weights::init_weights(this->weights, this->z.size());
	}
}