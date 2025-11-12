// TEAM-482: Noise schedule calculations
//
// Noise schedules control how noise is distributed across timesteps.
// Different schedules can dramatically affect image quality.
//
// Popular schedules:
// - Karras: Very popular for high-quality results (implement first!)
// - Exponential: Alternative high-quality schedule
// - Simple: Linear schedule, most compatible

use super::types::NoiseSchedule;

/// Calculate sigmas using Karras schedule
///
/// TEAM-482: Karras schedule is VERY popular in ComfyUI/A1111.
/// It produces high-quality results by concentrating more steps
/// in the important noise range.
///
/// Reference: https://arxiv.org/abs/2206.00364
///
/// # Arguments
/// * `num_steps` - Number of inference steps
/// * `sigma_min` - Minimum sigma value (default: 0.0292)
/// * `sigma_max` - Maximum sigma value (default: 14.6146)
/// * `rho` - Rho parameter for Karras schedule (default: 7.0)
///
/// # Returns
/// Vector of sigma values for each timestep
pub fn calculate_karras_sigmas(
    num_steps: usize,
    sigma_min: f64,
    sigma_max: f64,
    rho: f64,
) -> Vec<f64> {
    let min_inv_rho = sigma_min.powf(1.0 / rho);
    let max_inv_rho = sigma_max.powf(1.0 / rho);

    let mut sigmas: Vec<f64> = (0..num_steps)
        .map(|i| {
            let t = i as f64 / (num_steps - 1) as f64;
            let sigma = (max_inv_rho + t * (min_inv_rho - max_inv_rho)).powf(rho);
            sigma
        })
        .collect();

    // Add final sigma of 0.0
    sigmas.push(0.0);
    sigmas
}

/// Calculate sigmas using exponential schedule
///
/// TEAM-482: Exponential schedule is an alternative to Karras.
/// It provides smooth exponential decay of noise.
///
/// # Arguments
/// * `num_steps` - Number of inference steps
/// * `sigma_min` - Minimum sigma value
/// * `sigma_max` - Maximum sigma value
///
/// # Returns
/// Vector of sigma values for each timestep
pub fn calculate_exponential_sigmas(num_steps: usize, sigma_min: f64, sigma_max: f64) -> Vec<f64> {
    let mut sigmas: Vec<f64> = (0..num_steps)
        .map(|i| {
            let t = i as f64 / (num_steps - 1) as f64;
            let sigma = (sigma_max.ln() * (1.0 - t) + sigma_min.ln() * t).exp();
            sigma
        })
        .collect();

    // Add final sigma of 0.0
    sigmas.push(0.0);
    sigmas
}

/// Calculate sigmas using simple linear schedule
///
/// TEAM-482: Simple linear schedule is the most compatible.
/// It's the default for maximum compatibility.
///
/// # Arguments
/// * `num_steps` - Number of inference steps
/// * `sigma_min` - Minimum sigma value
/// * `sigma_max` - Maximum sigma value
///
/// # Returns
/// Vector of sigma values for each timestep
pub fn calculate_simple_sigmas(num_steps: usize, sigma_min: f64, sigma_max: f64) -> Vec<f64> {
    let mut sigmas: Vec<f64> = (0..num_steps)
        .map(|i| {
            let t = i as f64 / (num_steps - 1) as f64;
            sigma_max * (1.0 - t) + sigma_min * t
        })
        .collect();

    // Add final sigma of 0.0
    sigmas.push(0.0);
    sigmas
}

/// Calculate sigmas for a given noise schedule
///
/// TEAM-482: Main entry point for calculating sigmas.
/// Dispatches to the appropriate schedule function.
///
/// # Arguments
/// * `schedule` - The noise schedule to use
/// * `num_steps` - Number of inference steps
/// * `sigma_min` - Minimum sigma value (default: 0.0292)
/// * `sigma_max` - Maximum sigma value (default: 14.6146)
///
/// # Returns
/// Vector of sigma values for each timestep
pub fn calculate_sigmas(
    schedule: NoiseSchedule,
    num_steps: usize,
    sigma_min: f64,
    sigma_max: f64,
) -> Vec<f64> {
    match schedule {
        NoiseSchedule::Simple => calculate_simple_sigmas(num_steps, sigma_min, sigma_max),
        NoiseSchedule::Karras => calculate_karras_sigmas(num_steps, sigma_min, sigma_max, 7.0),
        NoiseSchedule::Exponential => calculate_exponential_sigmas(num_steps, sigma_min, sigma_max),
        NoiseSchedule::SgmUniform | NoiseSchedule::DdimUniform => {
            // TEAM-482: For now, use simple schedule for uniform schedules
            // TODO: Implement proper uniform schedules
            calculate_simple_sigmas(num_steps, sigma_min, sigma_max)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_karras_sigmas() {
        let sigmas = calculate_karras_sigmas(20, 0.0292, 14.6146, 7.0);
        assert_eq!(sigmas.len(), 21); // 20 steps + final 0.0
        assert!(sigmas[0] > sigmas[19]); // Decreasing
        assert_eq!(sigmas[20], 0.0); // Final sigma is 0.0
    }

    #[test]
    fn test_exponential_sigmas() {
        let sigmas = calculate_exponential_sigmas(20, 0.0292, 14.6146);
        assert_eq!(sigmas.len(), 21); // 20 steps + final 0.0
        assert!(sigmas[0] > sigmas[19]); // Decreasing
        assert_eq!(sigmas[20], 0.0); // Final sigma is 0.0
    }

    #[test]
    fn test_simple_sigmas() {
        let sigmas = calculate_simple_sigmas(20, 0.0292, 14.6146);
        assert_eq!(sigmas.len(), 21); // 20 steps + final 0.0
        assert!(sigmas[0] > sigmas[19]); // Decreasing
        assert_eq!(sigmas[20], 0.0); // Final sigma is 0.0
    }

    #[test]
    fn test_karras_different_from_simple() {
        let karras = calculate_karras_sigmas(20, 0.0292, 14.6146, 7.0);
        let simple = calculate_simple_sigmas(20, 0.0292, 14.6146);

        // Karras should produce different values than simple
        assert_ne!(karras[10], simple[10]);
    }

    #[test]
    fn test_calculate_sigmas_dispatch() {
        let karras = calculate_sigmas(NoiseSchedule::Karras, 20, 0.0292, 14.6146);
        let simple = calculate_sigmas(NoiseSchedule::Simple, 20, 0.0292, 14.6146);
        let exponential = calculate_sigmas(NoiseSchedule::Exponential, 20, 0.0292, 14.6146);

        assert_eq!(karras.len(), 21);
        assert_eq!(simple.len(), 21);
        assert_eq!(exponential.len(), 21);

        // All should be different
        assert_ne!(karras[10], simple[10]);
        assert_ne!(karras[10], exponential[10]);
        assert_ne!(simple[10], exponential[10]);
    }
}
