use crate::{Matrix, matrix::{d_relu, d_softmax_cross_entropy, mat_cross_entropy, mat_element_wise_mul, mat_mul, mat_relu, mat_softmax, mat_sub}};
const ETA: f32 = 1e-3;

#[derive(Debug)]
pub struct NeuralNetwork {
    weights: Matrix, // Weights for Input -> Hidden
    hidden: Matrix,  // Weights for Hidden -> Output
    biases: Matrix,  // Bias for Hidden layer
}

impl NeuralNetwork {
    pub fn new(input_dims: usize, output_dims: usize, hidden_dims: usize) -> Self {
        let mut weights = Matrix::new(hidden_dims, input_dims);
        weights.fill_random();

        let mut hidden = Matrix::new(output_dims, hidden_dims);
        hidden.fill_random();

        // Shape (1, hidden_dims) for broadcasting
        let mut biases = Matrix::new(1, hidden_dims);
        biases.fill_random();

        NeuralNetwork {
            hidden,
            weights,
            biases,
        }
    }

    pub fn forward(
        &mut self,
        x: &Matrix,
        y: Option<&Matrix>,
    ) -> Result<Matrix, String> {

        // 1. Layer 1: Linear (Input -> Hidden)
        // x: (Batch, In), weights: (Hidden, In) -> Transpose to (In, Hidden)
        let mut intermediate1 = mat_mul(x, &self.weights, false, true)
            .map_err(|e| e.to_string())?;

        // 2. Layer 1: Bias
        intermediate1.add_bias(&self.biases)?;

        // 3. Layer 1: Activation (ReLU)
        let intermediate2 = mat_relu(&intermediate1)
            .map_err(|e| e.to_string())?;

        // 4. Layer 2: Linear (Hidden -> output of hidden)
        // intermediate2: (Batch, Hidden), hidden: (Out, Hidden) -> Transpose to (Hidden, Out)
        let nn_out = mat_mul(&intermediate2, &self.hidden, false, true)
            .map_err(|e| e.to_string())?;

        let y_hat = mat_softmax(&nn_out).map_err(|e| e.to_string())
            .map_err(|e| e.to_string())?;


        match y {
            Some(y) => {
                // let loss = mat_cross_entropy(&y_hat, y).map_err(|e| e.to_string())?;

                // STEP 1: Error at Output (dZ2)
                // Shape: (Batch, Out)
                // dZ2 = Prediction - Target
                let d_z2 = d_softmax_cross_entropy(&y_hat, y).map_err(|e| e.to_string())?;

                // STEP 2: Gradients for Layer 2 Weights (dW2)
                // Shape: (Out, Hidden)
                // dW2 = dZ2.T * Activation1
                // We calculate the NOW but apply later (after we use weights for backprop)
                let mut d_hidden = mat_mul(&d_z2, &intermediate2, true, false).map_err(|e| e.to_string())?;

                // STEP 3: Propagate Error to Hidden Layer (dA1)
                // Shape: Batch , hidden
                // dA1 = dZ2 * Weights2
                // Note: weights are (out, hidden), so no transpose is needed for (batch, out)
                let d_a1 = mat_mul(&d_z2, &self.hidden, false, false).map_err(|e| e.to_string())?;

                // STEP 4: Application Activate Derivative (dZ1)
                // Shape: batch, hidden
                // dZ1 = dA1 * ReLU_Deriv(Z1)
                // We perform an element wise multiplication (Hadamard Product) manually
                let relu_mask = d_relu(&intermediate1).map_err(|e| e.to_string())?;
                let d_z1 = mat_element_wise_mul(&d_a1, &relu_mask).map_err(|e| e.to_string())?;

                // STEP 5: Gradients for Layer 1 Weights (dW1)
                // Shape: Hidden, In
                // dW1 = dZ1.T * input
                let mut d_weights = mat_mul(&d_z1, x, true, false).map_err(|e| e.to_string())?;

                let mut temp = Matrix::new(1, d_z1.rows);
                temp.mat_fill(1.0);
                let mut d_biases = mat_mul(&temp, &d_z1, false, false)  .map_err(|e| e.to_string())?;

                d_hidden.scale(ETA);
                d_weights.scale(ETA);
                d_biases.scale(ETA);

                self.hidden = mat_sub(&self.hidden, &d_hidden).map_err(|e| e.to_string())?;
                self.weights = mat_sub(&self.weights, &d_weights).map_err(|e| e.to_string())?;
                self.biases = mat_sub(&self.biases, &d_biases).map_err(|e| e.to_string())?;

                Ok(y_hat)
            }
            None => return Ok(y_hat)
        }

    }
}
