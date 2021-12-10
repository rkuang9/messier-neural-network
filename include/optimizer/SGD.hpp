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
		private:
			double learning_rate;

		public:
			explicit SGD(double learning_rate);

			void update(Matrix &weights, Matrix &nabla_weights) override;

            void update(ColumnVector &bias, ColumnVector &nabla_bias) override;
		};
	}
}

#endif //NEURALNETWORK_SGD_HPP
