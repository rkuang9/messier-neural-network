#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "include/common.hpp" // includes all required files

bool generate_boolean();

/**
 * Learns XOR truth table
 * 0 XOR 0 = 0
 * 1 XOR 1 = 0
 * 0 XOR 1 = 1
 * 1 XOR 1 = 1
 *
 * Change the operator in the variable ColumnVector output for other gate logic
 */

void Run()
{
    using namespace net;
    using namespace net::loss;
    using namespace net::activation;
    using namespace net::weights;
    using namespace net::optimizer;

    // set size of randomly generated training data
    unsigned sample_size = 10000;

    std::vector<ColumnVector> training_data, expected_output;

    training_data.reserve(sample_size);
    expected_output.reserve(sample_size);

    // generate testing data
    for (unsigned i = 0; i < sample_size; ++i) {
        double boolean_a = generate_boolean();
        double boolean_b = generate_boolean();

        ColumnVector input {{boolean_a, boolean_b}};
        training_data.push_back(input);

        ColumnVector output {{double(boolean_a != boolean_b)}}; // xor output
        expected_output.push_back(output);
    }

    Model<Quadratic> network {
        new Input<Sigmoid>(2),
        new Dense<Sigmoid, Xavier>(2, 4),
        new Dense<Sigmoid, Xavier>(4, 1),
    };

    // use standard stochastic gradient descent with a learning rate of 1
    optimizer::SGD opt(1);

    // train using the training data 100 times
    network.train(opt, 100, training_data, expected_output);



    bool test_loop = true;

    // use the predict function to feed forward a
    // ColumnVector of inputs and get the output layer result
    while (test_loop) {
        double a, b;
        std::cout << "input first boolean (0, 1):";
        std::cin >> a;
        std::cout << "input second boolean (0, 1):";
        std::cin >> b;
        std::cout << "\nexpected output: " << (a != b) << "\n";

        ColumnVector test {{a, b}};

        std::cout << "\npredicted xor output: " << network.predict(test);

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


bool generate_boolean()
{
    std::random_device                 random;
    std::mt19937                       mt(random());
    std::uniform_int_distribution<int> distribution(0, 1);

    return distribution(mt);
}