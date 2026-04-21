//! Radix-2 Cooley-Tukey FFT / IFFT for complex data.
//!
//! Used by [`surrogates`](super::surrogates) for phase randomization.
//! Not a general-purpose FFT library — covers the specific need of forward +
//! inverse transforms on power-of-two-padded series.

use std::f64::consts::PI;

/// A complex number stored as `(re, im)`.
pub type C64 = (f64, f64);

#[inline]
pub fn c_add(a: C64, b: C64) -> C64 {
    (a.0 + b.0, a.1 + b.1)
}

#[inline]
pub fn c_sub(a: C64, b: C64) -> C64 {
    (a.0 - b.0, a.1 - b.1)
}

#[inline]
pub fn c_mul(a: C64, b: C64) -> C64 {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Radix-2 in-place Cooley-Tukey FFT.
///
/// `data.len()` **must** be a power of two. `inverse = true` computes the
/// inverse DFT (with 1/N scaling applied).
pub fn fft(data: &mut [C64], inverse: bool) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "FFT requires power-of-two length");
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            data.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Butterfly passes.
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = sign * 2.0 * PI / len as f64;
        let w_n: C64 = (angle.cos(), angle.sin());

        let mut start = 0;
        while start < n {
            let mut w: C64 = (1.0, 0.0);
            for k in 0..half {
                let u = data[start + k];
                let t = c_mul(w, data[start + k + half]);
                data[start + k] = c_add(u, t);
                data[start + k + half] = c_sub(u, t);
                w = c_mul(w, w_n);
            }
            start += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for x in data.iter_mut() {
            x.0 *= scale;
            x.1 *= scale;
        }
    }
}

/// Next power of two ≥ `n`.
pub fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    n.next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn roundtrip_identity() {
        let original: Vec<C64> = (0..8).map(|i| (i as f64, 0.0)).collect();
        let mut data = original.clone();
        fft(&mut data, false);
        fft(&mut data, true);
        for (a, b) in data.iter().zip(original.iter()) {
            assert_relative_eq!(a.0, b.0, epsilon = 1e-10);
            assert_relative_eq!(a.1, b.1, epsilon = 1e-10);
        }
    }

    #[test]
    fn constant_signal_dc_only() {
        let mut data: Vec<C64> = vec![(5.0, 0.0); 4];
        fft(&mut data, false);
        // DC component = N * value
        assert_relative_eq!(data[0].0, 20.0, epsilon = 1e-10);
        for &(re, im) in &data[1..] {
            assert_relative_eq!(re, 0.0, epsilon = 1e-10);
            assert_relative_eq!(im, 0.0, epsilon = 1e-10);
        }
    }
}
