#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "include/common.hpp" // includes all required files

double generate_random_num(double minimum, double maximum);

/**
 * Learns adding positive numbers
 */

void Run()
{
    using namespace net;
    using namespace net::loss;
    using namespace net::activation;
    using namespace net::weights;
    using namespace net::optimizer;

    // set size of randomly generated training data
    unsigned sample_size = 1000;

    std::vector<ColumnVector> training_data, expected_output;

    training_data.reserve(sample_size);
    expected_output.reserve(sample_size);

    // generate testing data
    for (unsigned i = 0; i < sample_size; ++i) {
        double random_one   = generate_random_num(0, 1.0);
        double random_two   = generate_random_num(0, 1.0);
        double random_three = generate_random_num(0, 1.0);

        ColumnVector input {{random_one, random_two, random_three}};
        training_data.push_back(input);

        ColumnVector output {{random_one + random_two + random_three}};
        expected_output.push_back(output);
    }

    Model<Quadratic> network {
        new Input<ReLU>(3),
        new Dense<ReLU, Xavier>(3, 1),
        // addition can be learned without hidden layers, although it works with them
    };

    // use standard stochastic gradient descent with a learning rate of 0.01
    optimizer::SGD opt(0.01);

    // train using the training data 100 times
    network.train(opt, 100, training_data, expected_output);



    bool test_loop = true;

    // use the predict function to feed forward a
    // ColumnVector of inputs and get the output layer result
    while (test_loop) {
        double a, b, c;
        std::cout << "input number 1:";
        std::cin >> a;
        std::cout << "input number 2:";
        std::cin >> b;
        std::cout << "input number 3:";
        std::cin >> c;

        ColumnVector test {{a, b, c}};

        std::cout << "\npredicted sum: " << network.predict(test);

        std::cout << "\n\ncontinue? yes (1), no (0):";
        std::cin >> test_loop;
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


double generate_random_num(double minimum, double maximum)
{
    std::random_device                     random;
    std::mt19937                           mt(random());
    std::uniform_real_distribution<double> distribution(minimum, maximum);

    return distribution(mt);
}