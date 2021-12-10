cmake_minimum_required(VERSION 3.20)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 14)


set(SRC_FILES
        main.cpp
        include/enums.hpp
        include/typedefs.hpp
        include/common.hpp
        include/model.hpp src/model.ipp

        include/activation/sigmoid.hpp src/activation/sigmoid.cpp
        include/activation/tanh.hpp src/activation/tanh.cpp
        include/activation/relu.hpp src/activation/relu.cpp
        include/activation/leakyrelu.hpp src/activation/leakyrelu.cpp
        include/activation/softmax.hpp src/activation/softmax.cpp
        include/activation/swish.hpp src/activation/swish.cpp
        include/activation/softplus.hpp src/activation/softplus.cpp
        include/activation/softsign.hpp src/activation/softsign.cpp

        include/layer/layer.hpp
        include/layer/dense.hpp src/layer/dense.ipp
        include/layer/input.hpp src/layer/input.ipp

        include/weights/xavier.hpp src/weights/xavier.cpp
        include/weights/he.hpp src/weights/he.cpp
        include/weights/normal.hpp src/weights/normal.cpp

        include/loss/quadratic.hpp src/loss/quadratic.cpp
        include/loss/cross_entropy.hpp src/loss/cross_entropy.cpp
        include/loss/kldivergence.hpp src/loss/kldivergence.cpp
        include/loss/mean_absolute_error.hpp src/loss/mean_absolute_error.cpp

        include/optimizer/optimizer.hpp src/optimizer/optimizer.cpp
        include/optimizer/SGD.hpp src/optimizer/SGD.cpp
        include/optimizer/adam.hpp src/optimizer/adam.cpp
        )

#set(CMAKE_CXX_FLAGS "-O3 -DEIGEN_DONT_VECTORIZE -DNDEBUG")
set(CMAKE_CXX_FLAGS "-O2")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_compile_definitions("NN_DEBUG")

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(NeuralNetwork ${SRC_FILES})
