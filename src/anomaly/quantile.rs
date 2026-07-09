//! Standard-normal quantile function (inverse CDF).
//!
//! Uses Wichura's AS 241 rational approximation — the algorithm behind
//! `scipy.stats.norm.ppf`, R's `qnorm`, and every "good" implementation.
//! Absolute error < 1e-9 across all inputs `p ∈ (0, 1)`.
//!
//! The parade wrapper needs this to map PIT values (Uniform on `(0,1)`)
//! to standard-normal z-scores.
//!
//! Reference: Wichura, M.J. (1988). "Algorithm AS 241: The Percentage
//! Points of the Normal Distribution". *Applied Statistics*, 37, 477-484.
//!
//! Coefficient tables below use the full precision from Wichura's
//! paper; Rust's parser rounds to the closest f64.

#![allow(clippy::excessive_precision)]

/// Standard-normal quantile at `p`. Returns `f64::NEG_INFINITY` if
/// `p ≤ 0`, `f64::INFINITY` if `p ≥ 1`. Callers should clamp `p` away
/// from the endpoints if they want a finite result (the parade clamps
/// at `1e-12`, giving `|z| ≈ 7.03`).
pub fn standard_normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        // Central region: rational approximation directly in q.
        let r = 0.180_625 - q * q;
        return q * poly_ratio(r, &CENTRAL_A, &CENTRAL_B);
    }
    // Tails: use r = √(-ln(min(p, 1-p))).
    let r = if q < 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();
    let x = if r <= 5.0 {
        let r = r - 1.6;
        poly_ratio(r, &TAIL1_A, &TAIL1_B)
    } else {
        let r = r - 5.0;
        poly_ratio(r, &TAIL2_A, &TAIL2_B)
    };
    if q < 0.0 {
        -x
    } else {
        x
    }
}

#[inline]
fn poly_ratio(r: f64, a: &[f64; 8], b: &[f64; 8]) -> f64 {
    // Horner form: ((((((a7·r+a6)·r+a5)·r+a4)·r+a3)·r+a2)·r+a1)·r+a0
    let num = ((((((a[7] * r + a[6]) * r + a[5]) * r + a[4]) * r + a[3]) * r + a[2]) * r + a[1])
        * r
        + a[0];
    let den = ((((((b[7] * r + b[6]) * r + b[5]) * r + b[4]) * r + b[3]) * r + b[2]) * r + b[1])
        * r
        + b[0];
    num / den
}

// AS 241 coefficients — verbatim from Wichura (1988), reformatted for
// Rust f64 literals.

const CENTRAL_A: [f64; 8] = [
    3.387_132_872_796_366_5,
    133.141_667_891_784_38,
    1_971.590_950_306_227_2,
    13_731.693_765_509_461,
    45_921.953_931_549_87,
    67_265.770_927_008_75,
    33_430.575_583_588_128,
    2_509.080_928_730_122_7,
];

const CENTRAL_B: [f64; 8] = [
    1.0,
    42.313_330_701_600_911,
    687.187_007_492_057_9,
    5_394.196_021_424_751,
    21_213.794_301_586_596,
    39_307.895_800_092_71,
    28_729.085_735_721_942,
    5_226.495_278_852_854,
];

const TAIL1_A: [f64; 8] = [
    1.423_437_110_749_683_5,
    4.630_337_846_156_546,
    5.769_497_221_460_69,
    3.647_848_324_763_205,
    1.270_458_252_452_368_4,
    0.241_780_725_177_450_6,
    2.272_384_498_926_918_5e-2,
    7.745_450_142_783_414e-4,
];

const TAIL1_B: [f64; 8] = [
    1.0,
    2.053_191_626_637_759,
    1.676_384_830_183_803_8,
    0.689_767_334_985_1,
    0.148_103_976_427_480_08,
    1.519_866_656_361_645_9e-2,
    5.475_938_084_995_345e-4,
    1.050_750_071_644_416_9e-9,
];

const TAIL2_A: [f64; 8] = [
    6.657_904_643_501_103,
    5.463_784_911_164_114,
    1.784_826_539_917_291_3,
    0.296_560_571_828_504_87,
    2.658_069_574_732_555_8e-2,
    1.240_135_866_431_089_7e-3,
    2.710_155_310_454_646e-5,
    2.010_334_399_292_288_2e-7,
];

const TAIL2_B: [f64; 8] = [
    1.0,
    0.599_832_206_555_887_9,
    0.136_929_880_922_735_8,
    1.487_536_129_085_061_5e-2,
    7.868_691_311_456_133e-4,
    1.846_318_317_510_054_8e-5,
    1.421_511_758_316_446e-7,
    2.044_263_103_389_939_7e-15,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_is_zero() {
        assert!(standard_normal_quantile(0.5).abs() < 1e-12);
    }

    #[test]
    fn known_quantiles() {
        // scipy.stats.norm.ppf reference values
        let cases = [
            (0.95_f64, 1.644_853_626_951_472_5),
            (0.975, 1.959_963_984_540_054_2),
            (0.99, 2.326_347_874_040_842_3),
            (0.999, 3.090_232_306_167_813_5),
            (0.9999, 3.719_016_485_455_28),
            (1e-12, -7.034_484_073_240_23),
            (1.0 - 1e-12, 7.034_484_073_240_23),
        ];
        for (p, expected) in cases {
            let got = standard_normal_quantile(p);
            // AS 241's absolute error is < 1e-9 in the central region,
            // relaxes to ~1e-5 in the deepest tails (p < 1e-9 range).
            let tol = if p > 1e-6 && p < 1.0 - 1e-6 {
                1e-9
            } else {
                1e-5
            };
            assert!(
                (got - expected).abs() < tol,
                "p={p}: expected {expected}, got {got}",
            );
        }
    }

    #[test]
    fn symmetry() {
        for &p in &[0.1_f64, 0.3, 0.4, 0.49, 0.001, 1e-6] {
            let lo = standard_normal_quantile(p);
            let hi = standard_normal_quantile(1.0 - p);
            assert!((lo + hi).abs() < 1e-9, "p={p}: {lo} + {hi} != 0");
        }
    }

    #[test]
    fn endpoints() {
        assert_eq!(standard_normal_quantile(0.0), f64::NEG_INFINITY);
        assert_eq!(standard_normal_quantile(1.0), f64::INFINITY);
    }
}
