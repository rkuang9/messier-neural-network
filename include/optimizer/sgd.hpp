//
// Created by R on 11/17/2021.
//

#ifndef NEURALNETWORK_SGD_HPP
#define NEURALNETWORK_SGD_HPP

#include "include/typedefs.hpp"
#include "include/optimizer/optimizer.hpp"

namespace net
{
	namespace optimizer
	{
		class SGD : public Optimizer
		{
		public:
			explicit SGD(double learning_rate);

			void update(Matrix &weights, const Matrix &nabla_weights) override;

            void update(ColumnVector &bias, const ColumnVector &nabla_bias) override;

            double learning_rate = 0.1;

            //double epsilon_smoothing = 1e-8;
		};
	}
}

#endif //NEURALNETWORK_SGD_HPP
