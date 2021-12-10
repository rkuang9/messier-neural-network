//
// Created by R on 12/8/2021.
//

#include "include/loss/mean_absolute_error.hpp"

namespace net
{
    namespace loss
    {
        double MeanAbsoluteError::loss(ColumnVector &a, ColumnVector &expected)
        {
            return (a.array() - expected.array()).abs().sum() * a.rows();
        }


        ColumnVector MeanAbsoluteError::gradient(ColumnVector &a, ColumnVector &expected)
        {
            ColumnVector gradient;
            gradient.resize(a.rows());

            for (unsigned i = 0; i < gradient.size(); ++i) {
                if (a[i] > expected[i]) {
                    gradient[i] = -1;
                }
                else if (a[i] < expected[i]) {
                    gradient[i] = 1;
                }
                else {
                    // according to https://stats.stackexchange.com/questions/312737/mean-absolute-error-mae-derivative
                    // https://github.com/tensorflow/tensorflow/blob/c91e944d626b517781af6a63c0aee302ab2457e3/tensorflow/python/ops/math_grad.py#L577
                    // math_ops.sign(x) returns 0 at x = 0
                    // see https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/sign
                    gradient[i] = 0;
                }
            }
        }
    }
}