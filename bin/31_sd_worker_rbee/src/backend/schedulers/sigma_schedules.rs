// Sigma Schedule Implementations
//
// ✅ SHARED MODULE - Used by all schedulers
// Extracted from uni_pc.rs to eliminate duplication
//
// These schedules control how noise (sigma) varies across timesteps.
// Different schedules dramatically affect image quality.
//
// Reference: Candle stable_diffusion/uni_pc.rs:48-94

/// Sigma schedule variants
///
/// ✅ FULLY IMPLEMENTED - Both Karras and Exponential
#[derive(Debug, Clone, Copy)]
pub enum SigmaSchedule {
    Karras(KarrasSigmaSchedule),
    Exponential(ExponentialSigmaSchedule),
}

impl SigmaSchedule {
    /// Calculate sigma value at time t (0.0 to 1.0)
    ///
    /// ✅ FULLY IMPLEMENTED
    /// Reference: Candle uni_pc.rs:34-39
    pub fn sigma_t(&self, t: f64) -> f64 {
        match self {
            Self::Karras(schedule) => schedule.sigma_t(t),
            Self::Exponential(schedule) => schedule.sigma_t(t),
        }
    }
}

impl Default for SigmaSchedule {
    fn default() -> Self {
        Self::Karras(KarrasSigmaSchedule::default())
    }
}

/// Karras sigma schedule
///
/// ✅ FULLY IMPLEMENTED
/// Very popular for high-quality results. Concentrates more steps
/// in the important noise range.
///
/// Formula: sigma(t) = (sigma_max^(1/rho) + (1-t) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
///
/// Reference:
/// - Candle uni_pc.rs:48-73
/// - Paper: https://arxiv.org/abs/2206.00364
#[derive(Debug, Clone, Copy)]
pub struct KarrasSigmaSchedule {
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub rho: f64,
}

impl KarrasSigmaSchedule {
    /// Calculate sigma at time t using Karras formula
    ///
    /// ✅ FULLY IMPLEMENTED
    /// Reference: Candle uni_pc.rs:56-63
    pub fn sigma_t(&self, t: f64) -> f64 {
        let (min_inv_rho, max_inv_rho) =
            (self.sigma_min.powf(1.0 / self.rho), self.sigma_max.powf(1.0 / self.rho));

        (max_inv_rho + ((1.0 - t) * (min_inv_rho - max_inv_rho))).powf(self.rho)
    }

    /// Generate array of sigma values for discrete timesteps
    ///
    /// ✅ FULLY IMPLEMENTED
    /// This is the array-based version used by most schedulers
    /// Returns sigmas in DESCENDING order (high noise to low noise)
    pub fn sigmas_array(&self, num_steps: usize) -> Vec<f64> {
        let min_inv_rho = self.sigma_min.powf(1.0 / self.rho);
        let max_inv_rho = self.sigma_max.powf(1.0 / self.rho);
        
        let mut sigmas: Vec<f64> = (0..num_steps)
            .map(|i| {
                // Note: Karras formula already has (1-t), so we use t directly
                let t = i as f64 / (num_steps - 1) as f64;
                (max_inv_rho + t * (min_inv_rho - max_inv_rho)).powf(self.rho)
            })
            .collect();

        // Add final sigma of 0.0
        sigmas.push(0.0);
        sigmas
    }
}

impl Default for KarrasSigmaSchedule {
    fn default() -> Self {
        Self { sigma_max: 10.0, sigma_min: 0.1, rho: 4.0 }
    }
}

/// Exponential sigma schedule
///
/// ✅ FULLY IMPLEMENTED
/// Alternative to Karras. Provides smooth exponential decay of noise.
///
/// Formula: sigma(t) = exp(t * (ln(sigma_max) - ln(sigma_min)) + ln(sigma_min))
///
/// Reference: Candle uni_pc.rs:76-94
#[derive(Debug, Clone, Copy)]
pub struct ExponentialSigmaSchedule {
    pub sigma_min: f64,
    pub sigma_max: f64,
}

impl ExponentialSigmaSchedule {
    /// Calculate sigma at time t using exponential formula
    ///
    /// ✅ FULLY IMPLEMENTED
    /// Reference: Candle uni_pc.rs:83-85
    pub fn sigma_t(&self, t: f64) -> f64 {
        (t * (self.sigma_max.ln() - self.sigma_min.ln()) + self.sigma_min.ln()).exp()
    }

    /// Generate array of sigma values for discrete timesteps
    ///
    /// ✅ FULLY IMPLEMENTED
    /// This is the array-based version used by most schedulers
    /// Returns sigmas in DESCENDING order (high noise to low noise)
    pub fn sigmas_array(&self, num_steps: usize) -> Vec<f64> {
        let mut sigmas: Vec<f64> = (0..num_steps)
            .map(|i| {
                // Reverse: go from sigma_max (t=1) to sigma_min (t=0)
                let t = 1.0 - (i as f64 / (num_steps - 1) as f64);
                self.sigma_t(t)
            })
            .collect();

        // Add final sigma of 0.0
        sigmas.push(0.0);
        sigmas
    }
}

impl Default for ExponentialSigmaSchedule {
    fn default() -> Self {
        Self { sigma_max: 80.0, sigma_min: 0.1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_karras_schedule_defaults() {
        let schedule = KarrasSigmaSchedule::default();
        assert_eq!(schedule.sigma_min, 0.1);
        assert_eq!(schedule.sigma_max, 10.0);
        assert_eq!(schedule.rho, 4.0);
    }

    #[test]
    fn test_karras_sigma_calculation() {
        let schedule = KarrasSigmaSchedule::default();
        let sigma_start = schedule.sigma_t(0.0);
        let sigma_end = schedule.sigma_t(1.0);
        // At t=0, should be near sigma_min
        assert!((sigma_start - 0.1).abs() < 0.1);
        // At t=1, should be near sigma_max
        assert!((sigma_end - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_karras_sigmas_array() {
        let schedule = KarrasSigmaSchedule::default();
        let sigmas = schedule.sigmas_array(20);
        assert_eq!(sigmas.len(), 21); // 20 steps + final 0.0
        assert!(sigmas[0] > sigmas[19]); // Decreasing
        assert_eq!(sigmas[20], 0.0); // Final sigma is 0.0
    }

    #[test]
    fn test_exponential_schedule_defaults() {
        let schedule = ExponentialSigmaSchedule::default();
        assert_eq!(schedule.sigma_min, 0.1);
        assert_eq!(schedule.sigma_max, 80.0);
    }

    #[test]
    fn test_exponential_sigma_calculation() {
        let schedule = ExponentialSigmaSchedule::default();
        let sigma_start = schedule.sigma_t(0.0);
        let sigma_end = schedule.sigma_t(1.0);
        // At t=0, should be near sigma_min
        assert!((sigma_start - 0.1).abs() < 0.1);
        // At t=1, should be near sigma_max
        assert!((sigma_end - 80.0).abs() < 1.0);
    }

    #[test]
    fn test_exponential_sigmas_array() {
        let schedule = ExponentialSigmaSchedule::default();
        let sigmas = schedule.sigmas_array(20);
        assert_eq!(sigmas.len(), 21); // 20 steps + final 0.0
        assert!(sigmas[0] > sigmas[19]); // Decreasing
        assert_eq!(sigmas[20], 0.0); // Final sigma is 0.0
    }
}
