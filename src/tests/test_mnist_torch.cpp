#include <torch/torch.h>
#include <iostream>

// Define a simple MLP model
struct MLP : torch::nn::Module {
    MLP() {
        // Define the layers
        fc1 = register_module("fc1", torch::nn::Linear(784, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        // Define the forward pass
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/this->is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/this->is_training());
        x = fc3->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
    }

    // Layer definitions
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main() {
    // Set up device
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }

    // Create the MLP model and move it to the appropriate device
    MLP model;
    model.to(device);

    // Loss function and optimizer
    auto criterion = torch::nn::NLLLoss();
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.9));

    // Load dataset (using MNIST as an example)
    auto train_dataset = torch::data::datasets::MNIST("/home/jfa/projects/dml-from-scratch/src/tests/data/MNIST/raw")
                             .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                             .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_dataset), /*batch_size=*/64);

    // Training loop
    size_t num_epochs = 10;
    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        model.train();
        size_t batch_idx = 0;
        for (auto& batch : *train_loader) {
            // Transfer data to device
            auto data = batch.data.view({-1, 784}).to(device);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = model.forward(data);
            auto loss = criterion(output, target);

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            if (batch_idx++ % 10 == 0) {
                std::cout << "Train Epoch: " << epoch << " [" << batch_idx * batch.data.size(0)
                          << "/" << 60000 << "]"
                          << " Loss: " << loss.item<double>() << "\n";
            }
        }
    }

    return 0;
}
