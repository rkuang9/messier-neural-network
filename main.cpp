#include <iostream>
#include "include/common.hpp"
#include <vector>
#include <chrono>
#include <random>
#include <bits/stdc++.h>


double random_number(double minimum, double maximum)
{
    std::random_device                     random;
    std::mt19937                           mt(random());
    std::uniform_real_distribution<double> distribution(minimum, maximum);

    return distribution(mt);
}


int random_int(int minimum, int maximum)
{
    std::random_device                 random;
    std::mt19937                       mt(random());
    std::uniform_int_distribution<int> distribution(minimum, maximum);

    return distribution(mt);
}


void Run()
{
    using namespace net;
    using namespace net::optimizer;
    using namespace net::loss;
    using namespace net::activation;

    unsigned sample_size = 1000;

    std::vector<std::vector<double>> data;
    std::vector<std::vector<double>> expected;

    data.reserve(sample_size);
    expected.reserve(sample_size);

    for (unsigned i = 0; i < sample_size; ++i) {
        double random_one   = random_number(0, 1.0);
        double random_two   = random_number(0, 1.0);
        double random_three = random_number(0, 1.0);

        data.push_back(std::vector<double> {random_one, random_two, random_three});
        expected.push_back(std::vector<double> {random_one + random_two + random_three});
    }

    Model<Quadratic> network {
        new Input<activation::ReLU>(3),
        new Dense<activation::ReLU, Xavier>(3, 20, true),
        new Dense<activation::ReLU, Xavier>(20, 1, true),
    };

    optimizer::SGD opt(0.01);
    network.train(opt, 100, data, expected);

    while (true) {
        double a, b, c;
        std::cout << "input 1 number to sum:";
        std::cin >> a;
        std::cout << "input 2 number to sum:";
        std::cin >> b;
        std::cout << "input 3 number to sum:";
        std::cin >> c;

        ColumnVector test {
            {a},
            {b},
            {c},
        };

        std::cout << "\nprediction: " << network.predict(test) << "\n\n";
    }
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    Run();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms   = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::cout << "\n\n" << "Execution Time: " << ms.count() / 1e6 << " ms";

    return 0;
}