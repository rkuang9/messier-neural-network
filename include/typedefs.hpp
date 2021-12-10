//
// Created by R on 11/12/2021.
//

#ifndef NEURALNETWORK_TYPEDEFS_HPP
#define NEURALNETWORK_TYPEDEFS_HPP

#include <lib/Eigen/Dense>

namespace net
{
    typedef Eigen::VectorX<double> ColumnVector;
	//typedef Eigen::Matrix<double, 1, Eigen::Dynamic> ColumnVector;
    typedef Eigen::MatrixX<double> Matrix;

}

#endif //NEURALNETWORK_TYPEDEFS_HPP
