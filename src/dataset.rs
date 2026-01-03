use std::{fs::File, io::Read};

use safetensors::SafeTensors;

pub fn load_data() -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let mut file = File::open("data/train_test_data_mnist.safetensors")?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let safetensors = SafeTensors::deserialize(&buffer)?;
    let train_labels = safetensors.tensor("train_labels")?;
    let train_images = safetensors.tensor("train_images")?;
    let test_images = safetensors.tensor("test_images")?;
    let test_labels = safetensors.tensor("test_labels")?;

    let train_images_vec: Vec<f32> = match train_images.dtype() {
        safetensors::Dtype::F32 => {
            train_images
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        },
        _ => panic!("Unexpected dtype for train_images"),
    };

    let test_images_vec: Vec<f32> = match test_images.dtype() {
        safetensors::Dtype::F32 => {
            test_images
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        },
        _ => panic!("Unexpected dtype for test_images"),
    };

    // Labels are one-hot encoded i64 vectors, convert to f32 for matrix operations
    let test_labels_vec: Vec<f32> = match test_labels.dtype() {
        safetensors::Dtype::I64 => {
            test_labels
                .data()
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()) as f32)
                .collect()
        },
        _ => panic!("Unexpected dtype for test_labels"),
    };

    let train_labels_vec: Vec<f32> = match train_labels.dtype() {
        safetensors::Dtype::I64 => {
            train_labels
                .data()
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()) as f32)
                .collect()
        },
        _ => panic!("Unexpected dtype for train_labels"),
    };

    Ok((train_images_vec, train_labels_vec, test_images_vec, test_labels_vec))
}
