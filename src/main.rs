mod matrix;
mod dataset;
mod model;

use crate::{dataset::load_data, matrix::Matrix, model::NeuralNetwork};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load MNIST dataset
    println!("Loading dataset...");
    let (train_images, train_labels, test_images, test_labels) = load_data()?;

    println!("Train images length: {}", train_images.len());
    println!("Train labels length: {}", train_labels.len());

    // MNIST format: images are (N, 1, 28, 28), labels are (N, 10) one-hot encoded
    // Images: 60000 samples * 1 * 28 * 28 = 60000 * 784
    // Labels: 60000 samples * 10
    let input_dims = 784;  // 28 * 28
    let output_dims = 10;  // 10 classes (digits 0-9)
    let hidden_dims = 512;
    let batch_size = 32;
    let epochs = 10;

    // Calculate number of samples from labels (labels.len() / 10)
    let num_samples = train_labels.len() / output_dims;
    let num_batches = num_samples / batch_size;

    println!("Number of samples: {}", num_samples);
    println!("Number of batches: {}", num_batches);

    // Create neural network
    let mut nn = NeuralNetwork::new(input_dims, output_dims, hidden_dims);
    println!("Neural network initialized\n");

    // Calculate test set info
    let num_test_samples = test_labels.len() / output_dims;
    let num_test_batches = num_test_samples / batch_size;

    // Training loop
    for epoch in 0..epochs {
        println!("Epoch {}/{}", epoch + 1, epochs);

        // Training phase
        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;

            // Create batch input matrix (batch_size x 784)
            // Images are stored as flattened (N, 1, 28, 28) -> need to extract 784 values per sample
            let mut input = Matrix::new(batch_size, input_dims);
            for i in 0..batch_size {
                let sample_idx = start_idx + i;
                let image_start = sample_idx * input_dims;

                for j in 0..input_dims {
                    input.set(i, j, train_images[image_start + j]);
                }
            }

            // Create batch target matrix (batch_size x 10)
            // Labels are already one-hot encoded, stored as (N, 10)
            let mut target = Matrix::new(batch_size, output_dims);
            for i in 0..batch_size {
                let sample_idx = start_idx + i;
                let label_start = sample_idx * output_dims;

                for j in 0..output_dims {
                    target.set(i, j, train_labels[label_start + j]);
                }
            }

            // Forward pass with backpropagation
            match nn.forward(&input, Some(&target)) {
                Ok(_) => {},
                Err(e) => println!("Error in batch {}: {}", batch_idx, e)
            }

            if batch_idx % 100 == 0 {
                println!("  Training Batch {}/{}", batch_idx, num_batches);
            }
        }

        // Test phase - evaluate on test set
        println!("  Evaluating on test set...");
        let mut correct = 0;
        let mut total = 0;

        for batch_idx in 0..num_test_batches {
            let start_idx = batch_idx * batch_size;

            // Create batch input matrix for test data
            let mut test_input = Matrix::new(batch_size, input_dims);
            for i in 0..batch_size {
                let sample_idx = start_idx + i;
                let image_start = sample_idx * input_dims;

                for j in 0..input_dims {
                    test_input.set(i, j, test_images[image_start + j]);
                }
            }

            // Create batch target matrix for test data
            let mut test_target = Matrix::new(batch_size, output_dims);
            for i in 0..batch_size {
                let sample_idx = start_idx + i;
                let label_start = sample_idx * output_dims;

                for j in 0..output_dims {
                    test_target.set(i, j, test_labels[label_start + j]);
                }
            }

            // Forward pass without backpropagation (pass None for no training)
            match nn.forward(&test_input, None) {
                Ok(predictions) => {
                    // Calculate accuracy by comparing argmax of predictions vs targets
                    for i in 0..batch_size {
                        // Find predicted class (argmax of predictions)
                        let mut max_pred_val = -f32::INFINITY;
                        let mut pred_class = 0;
                        for j in 0..output_dims {
                            if let Some(val) = predictions.get(i, j) {
                                if *val > max_pred_val {
                                    max_pred_val = *val;
                                    pred_class = j;
                                }
                            }
                        }

                        // Find actual class (argmax of one-hot encoded target)
                        let mut max_target_val = -f32::INFINITY;
                        let mut actual_class = 0;
                        for j in 0..output_dims {
                            if let Some(val) = test_target.get(i, j) {
                                if *val > max_target_val {
                                    max_target_val = *val;
                                    actual_class = j;
                                }
                            }
                        }

                        if pred_class == actual_class {
                            correct += 1;
                        }
                        total += 1;
                    }
                },
                Err(e) => println!("Error in test batch {}: {}", batch_idx, e)
            }
        }

        let accuracy = (correct as f32 / total as f32) * 100.0;
        println!("  Test Accuracy: {}/{} ({:.2}%)\n", correct, total, accuracy);
    }

    println!("Training completed!");

    Ok(())
}
