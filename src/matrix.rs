use rand::Rng;
use rand_distr::StandardNormal;

const EPSILON: f32 = 1e-5;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {

    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&f32> {
        if row >= self.rows || col >= self.cols {
            return None;
        }

        let index = (row * self.cols) + col;
        self.data.get(index)
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        let index = (row * self.cols) + col;
        self.data[index] = value;
    }

    pub fn is_same_size(&self, other: &Matrix) -> bool {
        self.rows == other.rows && self.cols == other.cols
    }

    pub fn copy(&mut self, other: &Matrix) -> Result<(), String> {
        if !self.is_same_size(other) {
            return Err("Matrices have different sizes".to_string());
        }
        self.data.copy_from_slice(&other.data);
        Ok(())
    }

    pub fn mat_fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    pub fn scale(&mut self, scale: f32) {
        for ele in 0..self.data.len() {
            self.data[ele] = self.data[ele] * scale;
        }
    }

    pub fn add_scalar(&mut self, scalar: f32) {
        for ele in 0..self.data.len() {
            self.data[ele] = self.data[ele] + scalar;
        }
    }

    pub fn sum(&self) -> f32 {
        let mut sum = 0.0;
        for ele in 0..self.data.len() {
            sum += self.data[ele]
        }
        sum
    }

    pub fn fill_random(&mut self) {
        let mut rng = rand::rng();

        for ele in 0..self.data.len() {
            self.data[ele] = rng.sample(StandardNormal);
        }
    }

    // Adds a bias vector to every row of the matrix (Broadcasting)
    pub fn add_bias(&mut self, bias: &Matrix) -> Result<(), String> {
        // Validation: The bias must be a vector with length equal to our columns
        if bias.rows != 1 || bias.cols != self.cols {
            return Err(format!("Bias dimension mismatch: expected (1, {}), got ({}, {})", 
                self.cols, bias.rows, bias.cols));
        }

        // We assume your data is stored in a flat Vec<f32> in row-major order
        for row in 0..self.rows {
            for col in 0..self.cols {
                // Calculate the flat index for the current element
                let index = row * self.cols + col;
                
                // Add the corresponding bias value (from the single bias row)
                self.data[index] += bias.data[col];
            }
        }
        Ok(())
    }
}

pub fn equal(a: &f32, b: &f32) -> bool {
    if (a - b).abs() < EPSILON {
        return true;
    }
    false
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for row in 0..self.rows {
            for col in 0..self.cols {
                if let Some(value) = self.get(row, col) {
                    write!(f, "{:8.2} ", value)?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub fn mat_add(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    let mut out = Matrix::new(a.rows, a.cols);

    if !(a.is_same_size(b)) {
        return Err("Matrices must have the same dimensions".to_string());
    }

    for i in 0..out.data.len() {
        out.data[i] = a.data[i] + b.data[i];
    }

    Ok(out)
}

pub fn mat_sub(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    let mut out = Matrix::new(a.rows, a.cols);

    if !(a.is_same_size(b)) {
        return Err("Matrices must have the same dimensions".to_string());
    }

    for i in 0..out.data.len() {
        out.data[i] = a.data[i] - b.data[i];
    }

    Ok(out)
}
pub fn mat_mul(
    a: &Matrix,
    b: &Matrix,
    trans_a: bool,
    trans_b: bool,
) -> Result<Matrix, String> {
    // 1. Determine Logical Dimensions
    // If transposed, rows become cols and cols become rows
    let (rows_a, cols_a) = if trans_a {
        (a.cols, a.rows)
    } else {
        (a.rows, a.cols)
    };
    let (rows_b, cols_b) = if trans_b {
        (b.cols, b.rows)
    } else {
        (b.rows, b.cols)
    };

    // 2. Validate Dimensions using Logical sizes
    // Rule: Inner dimensions must match (A cols == B rows)
    // Rule: Output must match outer dimensions (A rows x B cols)
    if cols_a != rows_b {
        return Err(format!(
            "Dimension mismatch: Matrix A is {}x{} (inner {}), Matrix B is {}x{} (inner {})",
            rows_a, cols_a, cols_a, rows_b, cols_b, rows_b
        ));
    }

    let mut out = Matrix::new(rows_a, cols_b);

    // 3. Smart Accessor Closure
    // Calculates the physical index based on the transposition flag
    let get_val = |mat: &Matrix, r: usize, c: usize, transposed: bool| {
        let idx = if transposed {
            c * mat.cols + r // Transposed: treat (r,c) as (c,r)
        } else {
            r * mat.cols + c // Standard: row-major
        };
        mat.data[idx]
    };

    // 4. The Loop
    // Iterate over the OUTPUT dimensions (Logical Rows of A, Logical Cols of B)
    for i in 0..rows_a {
        for j in 0..cols_b {
            let mut sum = 0.0;
            // Dot product over the shared dimension (Logical Cols of A / Logical Rows of B)
            for k in 0..cols_a {
                let val_a = get_val(a, i, k, trans_a);
                let val_b = get_val(b, k, j, trans_b);
                sum += val_a * val_b;
            }
            out.data[i * out.cols + j] = sum;
        }
    }

    Ok(out)
}

pub fn mat_relu(input: &Matrix) -> Result<Matrix, String> {
    let mut output = Matrix::new(input.rows, input.cols);
    for i in 0..input.data.len() {
        output.data[i] = if input.data[i] > 0.0 {
            input.data[i]
        } else {
            0.0
        };
    }

    Ok(output)
}

pub fn mat_softmax(input: &Matrix) -> Result<Matrix, String> {
    let mut output = Matrix::new(input.rows, input.cols);

    for row in 0..input.rows {
        // Optional: Numerical Stability fix (subtract max)
        // prevents exp() from overflowing to infinity
        let mut max_val = f32::MIN;
        for col in 0..input.cols {
            if let Some(val) = input.get(row, col) {
                if *val > max_val { max_val = *val; }
            }
        }

        let mut row_sum = 0.0;
        
        // Exponentiate
        for col in 0..input.cols {
            if let Some(val) = input.get(row, col) {
                let e = (val - max_val).exp(); // subtract max for stability
                output.set(row, col, e);
                row_sum += e;
            }
        }

        // Normalize (Divide, do not scale by sum)
        for col in 0..input.cols {
            if let Some(val) = output.get(row, col) {
                output.set(row, col, val / row_sum);
            }
        }
    }
    Ok(output)
}

pub fn d_softmax_cross_entropy(z: &Matrix, y: &Matrix) -> Result<Matrix, String> {
    if !z.is_same_size(y) {
        return Err("Prediction (z) and Targets (y) must have the same dimensions".to_string());
    }

    let n = z.rows as f32;

    let mut grad = Matrix::new(z.rows, z.cols);

    for i in 0..z.data.len() {
        let pred = z.data[i];
        let target = y.data[i];
        grad.data[i] = (pred - target) / n;
    }

    Ok(grad)
}

pub fn d_relu(input: &Matrix) -> Result<Matrix, String> {
    let mut output = Matrix::new(input.rows, input.cols);
    for i in 0..input.data.len() {
        output.data[i] = if input.data[i] > 0.0 {
            1.0 
        } else {
            0.0
        };
    }
    Ok(output)
}

pub fn mat_cross_entropy(y_hat: &Matrix, y: &Matrix) -> Result<f32, String> {
    // sum(y log y - (1 - y) log (1 - y))
    if !y_hat.is_same_size(y) {
        return Err(format!(
            "Dimension mismatch: Matrix y_hat is {}x{}, Matrix y is {}x{}",
            y_hat.rows, y_hat.cols, y.rows, y.cols,
        ));
    }

    let mut total_loss = 0.0;
    
    for i in 0..y_hat.data.len() {
        let pred = y_hat.data[i].clamp(EPSILON, 1.0 - EPSILON);
        let actual = y.data[i];
        // Calculate Binary Cross Entropy
        total_loss += actual * pred.ln() + (1.0 - actual) * (1.0 - pred).ln();
    }

    // Average the loss (negative sign is standard for NLL)
    Ok(-(total_loss / y_hat.data.len() as f32))
}


pub fn mat_element_wise_mul(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if !a.is_same_size(b) {
        return Err(format!(
            "Dimension mismatch: Matrix a is {}x{}, Matrix b is {}x{}",
            a.rows, a.cols, b.rows, b.cols,
        ));
    }
    let mut output = Matrix::new(a.rows, a.cols);

    for ele in 0..a.data.len() {
        output.data[ele] = a.data[ele] * b.data[ele];
    }

    Ok(output)

}
