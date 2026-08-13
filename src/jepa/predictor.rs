// src/jepa/predictor.rs — Next-state prediction
//
// In v1: a linear predictor that maps current embedding to predicted next-bar embedding.
// In v2+: a transformer-based predictor.

use super::embedding::{Embedding, EMBEDDING_DIM};

/// A simple linear predictor for next-bar embedding.
///
/// Predicts: emb_{t+1} = W @ emb_t + b
/// where W is a EMBEDDING_DIM × EMBEDDING_DIM matrix (256×256).
pub struct LinearPredictor {
    /// Weight matrix (row-major, EMBEDDING_DIM × EMBEDDING_DIM).
    weights: Box<[[f32; EMBEDDING_DIM]; EMBEDDING_DIM]>,
    /// Bias vector.
    bias: [f32; EMBEDDING_DIM],
}

impl LinearPredictor {
    /// Create a predictor initialized to identity (predicts "same as current").
    pub fn new() -> Self {
        let mut weights = Box::new([[0.0f32; EMBEDDING_DIM]; EMBEDDING_DIM]);
        for i in 0..EMBEDDING_DIM {
            weights[i][i] = 1.0;
        }
        Self {
            weights,
            bias: [0.0; EMBEDDING_DIM],
        }
    }

    /// Predict the next-bar embedding from the current embedding.
    pub fn predict(&self, current: &Embedding) -> Embedding {
        let mut out = self.bias;
        for i in 0..EMBEDDING_DIM {
            let mut sum = self.bias[i];
            for j in 0..EMBEDDING_DIM {
                sum += self.weights[i][j] * current[j];
            }
            out[i] = sum;
        }
        out
    }

    /// Compute cosine prediction error between a prediction and actual embedding.
    ///
    /// Returns cosine distance: 0 = identical direction, 1 = orthogonal, 2 = opposite.
    /// This aligns with fleet-ensemble's `cosine_distance` metric.
    pub fn prediction_error(predicted: &Embedding, actual: &Embedding) -> f32 {
        cosine_distance(predicted, actual)
    }

    /// Update weights using a simple gradient step.
    /// learning_rate * (actual - predicted) for each dimension.
    /// This is a crude online update, not a proper optimizer, but
    /// sufficient for v1's curiosity harness.
    pub fn online_update(&mut self, current: &Embedding, predicted: &Embedding, actual: &Embedding, lr: f32) {
        let error: [f32; EMBEDDING_DIM] = std::array::from_fn(|i| actual[i] - predicted[i]);
        for i in 0..EMBEDDING_DIM {
            for j in 0..EMBEDDING_DIM {
                // dW[i][j] = error[i] * current[j]
                self.weights[i][j] += lr * error[i] * current[j];
            }
            self.bias[i] += lr * error[i];
        }
    }

    /// Get a reference to the weights (for serialization/inspection).
    pub fn weights(&self) -> &[[f32; EMBEDDING_DIM]; EMBEDDING_DIM] {
        &self.weights
    }
}

/// Cosine distance between two embedding vectors: 0 = identical, 1 = orthogonal.
///
/// Consistent with fleet-ensemble's `cosine_distance` function.
#[inline]
fn cosine_distance(a: &Embedding, b: &Embedding) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..EMBEDDING_DIM {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-10);
    1.0 - (dot / denom)
}

impl Default for LinearPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_prediction() {
        let predictor = LinearPredictor::new();
        let input = [0.5; EMBEDDING_DIM];
        let output = predictor.predict(&input);
        // Identity weights should produce the same vector
        for i in 0..EMBEDDING_DIM {
            assert_approx_eq::assert_approx_eq!(output[i], input[i], 1e-6);
        }
    }

    #[test]
    fn test_prediction_error_zero() {
        let a = [0.3, 0.7, 0.5, 0.1, 0.9, 0.4, 0.6, 0.2,
                 0.8, 0.55, 0.45, 0.35, 0.65, 0.25, 0.75, 0.15];
        // Pad to full EMBEDDING_DIM
        let mut a_full = [0.0f32; EMBEDDING_DIM];
        a_full[..16].copy_from_slice(&a);
        // Cosine distance of a vector with itself = 0 (within float epsilon)
        let err = LinearPredictor::prediction_error(&a_full, &a_full);
        assert!(err.abs() < 1e-5, "expected ~0, got {err}");
    }

    #[test]
    fn test_prediction_error_nonzero() {
        let a = [1.0; EMBEDDING_DIM];
        let b = [0.0; EMBEDDING_DIM];
        let err = LinearPredictor::prediction_error(&a, &b);
        // Cosine distance of [1.0; N] and [0.0; N]: zero vector has zero norm,
        // denom clamped to 1e-10, so distance = 1.0 - 0.0/N = 1.0
        assert_approx_eq::assert_approx_eq!(err, 1.0, 0.001);
    }

    #[test]
    fn test_prediction_error_orthogonal() {
        let mut a = [0.0f32; EMBEDDING_DIM];
        let mut b = [0.0f32; EMBEDDING_DIM];
        a[0] = 1.0;
        b[1] = 1.0;
        let err = LinearPredictor::prediction_error(&a, &b);
        // Orthogonal vectors: cosine distance = 1.0
        assert_approx_eq::assert_approx_eq!(err, 1.0, 1e-5);
    }

    #[test]
    fn test_online_update_reduces_error() {
        let mut predictor = LinearPredictor::new();
        // Use vectors with different directions so cosine distance is non-zero
        let mut current = [0.0f32; EMBEDDING_DIM];
        let mut actual = [0.0f32; EMBEDDING_DIM];
        current[0] = 1.0;
        current[1] = 0.5;
        actual[0] = 0.5;
        actual[1] = 1.0;

        let predicted_before = predictor.predict(&current);
        let err_before = LinearPredictor::prediction_error(&predicted_before, &actual);
        assert!(err_before > 0.0, "setup should have non-zero initial error");

        // Train for 100 steps
        for _ in 0..100 {
            let predicted = predictor.predict(&current);
            predictor.online_update(&current, &predicted, &actual, 0.1);
        }

        let predicted_after = predictor.predict(&current);
        let err_after = LinearPredictor::prediction_error(&predicted_after, &actual);

        assert!(err_after < err_before, "error should decrease after training: before={err_before}, after={err_after}");
    }
}
