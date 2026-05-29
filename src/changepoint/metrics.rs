//! Evaluation metrics for changepoint segmentations.
//!
//! Mirrors `ruptures.metrics`:
//!
//! - [`precision_recall`] — predicted-vs-true match within a margin.
//! - [`hausdorff`] — max one-sided distance between two breakpoint sets.
//! - [`randindex`] — agreement of two segmentations as set partitions.
//!
//! Conventions follow ruptures: a breakpoint list is sorted and
//! includes the terminal `n` (signal length). Internal changepoints
//! are everything except that terminal element.

use crate::error::{ForecastError, Result};

/// Precision / recall / F1 result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrecisionRecall {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// Compute precision / recall / F1 of a predicted breakpoint list
/// against the ground truth.
///
/// A predicted breakpoint is considered matched if it falls within
/// `margin` of *some* true breakpoint (and each true breakpoint can
/// only be matched once — greedy left-to-right).
///
/// `bkps_true` and `bkps_pred` are expected in ruptures convention
/// (sorted, with terminal `n` included). The terminal `n` is excluded
/// from both sides before scoring — it isn't a "detected" event.
pub fn precision_recall(
    bkps_true: &[usize],
    bkps_pred: &[usize],
    margin: usize,
) -> Result<PrecisionRecall> {
    let truth = drop_terminal(bkps_true);
    let pred = drop_terminal(bkps_pred);
    if pred.is_empty() && truth.is_empty() {
        return Ok(PrecisionRecall {
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
        });
    }
    let mut matched_truth = vec![false; truth.len()];
    let mut tp = 0usize;
    for &p in &pred {
        let mut best_i: Option<usize> = None;
        let mut best_dist = usize::MAX;
        for (i, &t) in truth.iter().enumerate() {
            if matched_truth[i] {
                continue;
            }
            let dist = p.abs_diff(t);
            if dist <= margin && dist < best_dist {
                best_dist = dist;
                best_i = Some(i);
            }
        }
        if let Some(i) = best_i {
            matched_truth[i] = true;
            tp += 1;
        }
    }
    let precision = if pred.is_empty() {
        1.0
    } else {
        tp as f64 / pred.len() as f64
    };
    let recall = if truth.is_empty() {
        1.0
    } else {
        tp as f64 / truth.len() as f64
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    Ok(PrecisionRecall {
        precision,
        recall,
        f1,
    })
}

/// Hausdorff distance between two breakpoint lists.
///
/// `H(A, B) = max(max_{a in A} min_{b in B} |a - b|, max_{b in B} min_{a in A} |a - b|)`.
///
/// Returns 0 when both lists are empty.
pub fn hausdorff(bkps1: &[usize], bkps2: &[usize]) -> Result<f64> {
    let a = drop_terminal(bkps1);
    let b = drop_terminal(bkps2);
    if a.is_empty() && b.is_empty() {
        return Ok(0.0);
    }
    if a.is_empty() || b.is_empty() {
        return Err(ForecastError::InvalidParameter(
            "hausdorff: one of the breakpoint lists has zero internal changepoints; \
             distance is undefined"
                .into(),
        ));
    }
    let one_sided = |xs: &[usize], ys: &[usize]| -> f64 {
        xs.iter()
            .map(|&x| {
                ys.iter()
                    .map(|&y| x.abs_diff(y) as f64)
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0_f64, f64::max)
    };
    Ok(one_sided(&a, &b).max(one_sided(&b, &a)))
}

/// Rand index of two segmentations.
///
/// Treats each segmentation as a partition of `{0, …, n-1}` and counts
/// the fraction of index pairs that agree (both same segment, or both
/// different segment) in `(bkps_true, bkps_pred)`. Returns a value in
/// `[0, 1]`; 1 means identical segmentations.
///
/// `n` must equal the terminal element of *both* breakpoint lists.
pub fn randindex(bkps_true: &[usize], bkps_pred: &[usize], n: usize) -> Result<f64> {
    if n == 0 {
        return Ok(1.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    // Build per-index segment labels.
    let labels_true = labels_from_bkps(bkps_true, n)?;
    let labels_pred = labels_from_bkps(bkps_pred, n)?;

    let mut agreeing = 0_u64;
    let total = (n * (n - 1) / 2) as u64;
    for i in 0..n {
        for j in (i + 1)..n {
            let same_true = labels_true[i] == labels_true[j];
            let same_pred = labels_pred[i] == labels_pred[j];
            if same_true == same_pred {
                agreeing += 1;
            }
        }
    }
    Ok(agreeing as f64 / total as f64)
}

fn drop_terminal(bkps: &[usize]) -> Vec<usize> {
    if bkps.is_empty() {
        return Vec::new();
    }
    bkps[..bkps.len() - 1].to_vec()
}

fn labels_from_bkps(bkps: &[usize], n: usize) -> Result<Vec<usize>> {
    if bkps.is_empty() {
        return Err(ForecastError::InvalidParameter(
            "labels_from_bkps: breakpoint list cannot be empty".into(),
        ));
    }
    if *bkps.last().unwrap() != n {
        return Err(ForecastError::InvalidParameter(format!(
            "labels_from_bkps: terminal breakpoint ({}) must equal n ({})",
            bkps.last().unwrap(),
            n
        )));
    }
    let mut labels = vec![0usize; n];
    let mut start = 0usize;
    for (label, &end) in bkps.iter().enumerate() {
        for k in start..end.min(n) {
            labels[k] = label;
        }
        start = end;
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn precision_recall_exact_match_is_perfect() {
        let truth = vec![30, 60, 90, 100];
        let pred = vec![30, 60, 90, 100];
        let pr = precision_recall(&truth, &pred, 0).unwrap();
        assert_relative_eq!(pr.precision, 1.0, epsilon = 1e-12);
        assert_relative_eq!(pr.recall, 1.0, epsilon = 1e-12);
        assert_relative_eq!(pr.f1, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn precision_recall_within_margin_matches() {
        let truth = vec![30, 60, 100];
        let pred = vec![28, 64, 100];
        let pr = precision_recall(&truth, &pred, 5).unwrap();
        assert_relative_eq!(pr.precision, 1.0, epsilon = 1e-12);
        assert_relative_eq!(pr.recall, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn precision_recall_missed_changepoint_penalises_recall() {
        let truth = vec![30, 60, 100];
        let pred = vec![30, 100];
        let pr = precision_recall(&truth, &pred, 2).unwrap();
        assert_relative_eq!(pr.precision, 1.0, epsilon = 1e-12);
        assert_relative_eq!(pr.recall, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn precision_recall_no_truth_no_pred_perfect() {
        let pr = precision_recall(&[100], &[100], 5).unwrap();
        assert_relative_eq!(pr.precision, 1.0, epsilon = 1e-12);
        assert_relative_eq!(pr.recall, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn hausdorff_exact_match_is_zero() {
        let a = vec![30, 60, 100];
        let b = vec![30, 60, 100];
        assert_relative_eq!(hausdorff(&a, &b).unwrap(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn hausdorff_one_sided_max() {
        // truth = [30, 60, 100]; pred = [25, 100] → internal [25] vs [30, 60].
        // h([30, 60], [25]) = max(min(|30-25|), min(|60-25|)) = max(5, 35) = 35
        // h([25], [30, 60]) = min(|25-30|, |25-60|) = 5
        // overall = 35
        let a = vec![30, 60, 100];
        let b = vec![25, 100];
        assert_relative_eq!(hausdorff(&a, &b).unwrap(), 35.0, epsilon = 1e-12);
    }

    #[test]
    fn randindex_identical_is_one() {
        let bkps = vec![30, 60, 100];
        assert_relative_eq!(randindex(&bkps, &bkps, 100).unwrap(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn randindex_off_by_one_is_high_not_perfect() {
        // truth [50, 100], pred [49, 100]: one index labelled differently.
        let truth = vec![50, 100];
        let pred = vec![49, 100];
        let r = randindex(&truth, &pred, 100).unwrap();
        assert!(r < 1.0 && r > 0.95);
    }

    #[test]
    fn randindex_inverted_is_zero_only_at_degenerate() {
        // 3 segments vs 1 segment: different partitions.
        let truth = vec![3, 6, 9];
        let pred = vec![9];
        let r = randindex(&truth, &pred, 9).unwrap();
        // Some pairs still agree (the cross-segment pairs in truth are
        // forced to "different" in truth but "same" in pred → disagree),
        // but the within-segment pairs (same in truth, same in pred) agree.
        // 3 in-segment + 0 different-in-both = 3 / C(9, 2) = 3/36 = 0.0833.
        assert!(r < 0.3);
    }
}
