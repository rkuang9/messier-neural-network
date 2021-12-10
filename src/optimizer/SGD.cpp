//
// Created by R on 11/17/2021.
//

#include "include/optimizer/SGD.hpp"

namespace net
{
	namespace optimizer
	{
		SGD::SGD(double learning_rate) : learning_rate(learning_rate)
		{
		}


		void SGD::update(Matrix &weights, Matrix &nabla_weights)
		{
			weights -= (this->learning_rate * nabla_weights);
		}


        void SGD::update(ColumnVector &bias, ColumnVector &nabla_bias)
        {
            bias -= (this->learning_rate * nabla_bias);
        }
	}
}