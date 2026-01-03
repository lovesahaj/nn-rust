# PyTorch-Rust: Neural Network from Scratch

A pure Rust implementation of a neural network for MNIST digit classification, built entirely from scratch without using external ML frameworks.

## Features

- **Custom Matrix Operations**: Hand-written linear algebra library with BLAS-like operations
- **Neural Network**: Two-layer fully connected network with backpropagation
- **MNIST Classification**: Achieves **88%+ accuracy** on handwritten digit recognition
- **Zero ML Dependencies**: Only uses basic Rust libraries (no PyTorch, TensorFlow, or similar)
- **Educational**: Clean, readable code perfect for understanding neural networks from first principles

## Architecture

```
Input Layer (784)  →  Hidden Layer (128, ReLU)  →  Output Layer (10, Softmax)
```

**Model Details:**

- Input: 28x28 grayscale images flattened to 784 features
- Hidden layer: 128 neurons with ReLU activation
- Output layer: 10 classes (digits 0-9) with Softmax activation
- Loss: Cross-entropy
- Optimizer: Stochastic Gradient Descent (SGD)
- Learning rate: 1e-4
- Batch size: 32

## Results

Training on 60,000 MNIST samples over 10 epochs:

| Epoch | Test Accuracy |
| ----- | ------------- |
| 1     | 73.31%        |
| 2     | 79.67%        |
| 3     | 82.72%        |
| 4     | 84.35%        |
| 5     | 85.50%        |
| 6     | 86.23%        |
| 7     | 86.83%        |
| 8     | 87.30%        |
| 9     | 87.72%        |
| 10    | **88.03%**    |

## Project Structure

```
pytorch-rust/
├── src/
│   ├── main.rs       # Training loop and evaluation
│   ├── matrix.rs     # Matrix operations and linear algebra
│   ├── model.rs      # Neural network implementation
│   └── dataset.rs    # MNIST data loader
├── data/             # MNIST dataset (SafeTensors format)
└── Cargo.toml        # Dependencies
```

## Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- MNIST dataset in SafeTensors format (place in `data/` directory)

## Installation

```bash
# Clone the repository
git clone https://github.com/lovesahaj/nn-rust
cd pytorch-rust

# Build the project
cargo build --release

# Run training
cargo run --release
```

## Usage

Run the training:

```bash
cargo run --release
```

The program will:

1. Load the MNIST training and test datasets
2. Initialize a neural network with random weights
3. Train for 10 epochs with mini-batch gradient descent
4. Evaluate accuracy on the test set after each epoch
5. Display final accuracy

## Implementation Highlights

### Matrix Operations (`matrix.rs`)

- Matrix multiplication with transpose support
- Element-wise operations (add, subtract, multiply, scale)
- Activation functions (ReLU, Softmax)
- Loss functions (Cross-entropy)
- Gradient computation for backpropagation

### Neural Network (`model.rs`)

- Forward propagation with cached intermediate values
- Backpropagation using chain rule
- Gradient descent weight updates
- Proper weight initialization (standard normal distribution)

### Training Loop (`main.rs`)

- Mini-batch processing
- Per-epoch evaluation on test set
- Progress tracking with batch counters

## Dependencies

```toml
rand = "0.9.2"           # Random number generation
rand_distr = "0.5.1"     # Statistical distributions
safetensors = "0.7.0"    # Safe tensor file format
```

## Performance Notes

**Current Configuration:**

- Batch size: 32
- Training time: ~minutes per epoch (CPU)
- Memory usage: Minimal

**Potential Optimizations:**

- Use BLAS libraries (ndarray + openblas) for 5-20x speedup
- Increase batch size to 128+ for better throughput
- Enable parallel processing with Rayon
- Build with `--release` flag (10-50x faster than debug mode)

## Learning Resources

This project demonstrates:

- **Linear Algebra**: Matrix operations and transformations
- **Calculus**: Gradient descent and backpropagation
- **Rust**: Ownership, error handling, and performance optimization
- **Machine Learning**: Neural network fundamentals

## Limitations

- No GPU acceleration (CPU only)
- Basic optimizer (no Adam, momentum, etc.)
- No regularization (dropout, L2, etc.)
- Single hidden layer architecture
- No model saving/loading

---

**Built with ❤️ and Rust**
