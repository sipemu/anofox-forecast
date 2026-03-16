//! Unified report builder for pipeline results.
//!
//! [`PipelineReport`] produces a structured, multi-section report from a
//! [`PipelineResult`] that can be displayed as formatted text or stored
//! via the [`Storable`](super::store::Storable) trait.

use std::fmt;

use super::pipeline::PipelineResult;

/// A structured report section.
#[derive(Debug, Clone)]
pub struct ReportSection {
    pub heading: String,
    pub content: ReportContent,
}

/// Content within a report section.
#[derive(Debug, Clone)]
pub enum ReportContent {
    /// Free-form text.
    Text(String),
    /// Key-value pairs.
    KeyValue(Vec<(String, String)>),
    /// Table with headers and rows.
    Table {
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
    },
}

/// A full pipeline report with multiple sections.
#[derive(Debug, Clone)]
pub struct PipelineReport {
    pub title: String,
    pub sections: Vec<ReportSection>,
}

impl PipelineReport {
    /// Build a report from a pipeline result.
    pub fn from_result(result: &PipelineResult) -> Self {
        let mut sections = Vec::new();

        // 1. Summary
        sections.push(ReportSection {
            heading: "Summary".into(),
            content: ReportContent::KeyValue(vec![
                ("Model".into(), result.model_name.clone()),
                (
                    "Horizon".into(),
                    format!("{} steps", result.forecast.primary().len()),
                ),
            ]),
        });

        // 1b. Structural break warning
        if result.structural_break_in_holdout {
            sections.push(ReportSection {
                heading: "Warnings".into(),
                content: ReportContent::Text(
                    "Structural break detected in holdout period. Accuracy metrics \
                     may be unreliable. Consider excluding from aggregated summaries."
                        .into(),
                ),
            });
        }

        // 2. Data Profile
        if let Some(ref profile) = result.profile {
            sections.push(ReportSection {
                heading: "Data Profile".into(),
                content: ReportContent::KeyValue(vec![
                    ("Observations".into(), format!("{}", profile.n_observations)),
                    ("Mean".into(), format!("{:.4}", profile.mean)),
                    ("Std Dev".into(), format!("{:.4}", profile.std_dev)),
                    ("Trend".into(), format!("{}", profile.trend_direction)),
                    ("Stationary".into(), format!("{}", profile.is_stationary())),
                    (
                        "Intermittent".into(),
                        format!("{}", profile.is_intermittent),
                    ),
                    (
                        "Quality Score".into(),
                        format!("{:.2}", profile.quality_score),
                    ),
                ]),
            });
        }

        // 3. Preprocessing
        if let Some(ref pp) = result.preprocess {
            if !pp.steps_applied.is_empty() {
                let mut kv = Vec::new();
                if let Some(lambda) = pp.boxcox_lambda {
                    kv.push(("Box-Cox Lambda".into(), format!("{:.4}", lambda)));
                }
                if pp.outliers_replaced > 0 {
                    kv.push((
                        "Outliers Replaced".into(),
                        format!("{}", pp.outliers_replaced),
                    ));
                }
                sections.push(ReportSection {
                    heading: "Preprocessing".into(),
                    content: ReportContent::KeyValue(kv),
                });
            }
        }

        // 4a. Trend/Seasonal Selection
        if let Some(ref ts) = result.trend_selection {
            let rows: Vec<Vec<String>> = ts
                .scores
                .iter()
                .map(|(name, score)| {
                    let marker = if *name == ts.selected { " *" } else { "" };
                    vec![format!("{}{}", name, marker), format!("{:.4}", score)]
                })
                .collect();
            sections.push(ReportSection {
                heading: "Trend Selection".into(),
                content: ReportContent::Table {
                    headers: vec!["Component".into(), ts.criterion.clone()],
                    rows,
                },
            });
        }
        if let Some(ref ss) = result.seasonal_selection {
            let rows: Vec<Vec<String>> = ss
                .scores
                .iter()
                .map(|(name, score)| {
                    let marker = if *name == ss.selected { " *" } else { "" };
                    vec![format!("{}{}", name, marker), format!("{:.4}", score)]
                })
                .collect();
            sections.push(ReportSection {
                heading: "Seasonal Selection".into(),
                content: ReportContent::Table {
                    headers: vec!["Component".into(), ss.criterion.clone()],
                    rows,
                },
            });
        }

        // 4. Model Selection
        {
            let mut kv = Vec::new();
            if let Some(ref qf) = result.quality_floor {
                kv.push((
                    "Quality Floor".into(),
                    format!(
                        "{} (SPA p={:.4})",
                        if qf.is_outperformed {
                            "PASSED"
                        } else {
                            "FAILED"
                        },
                        qf.spa_p_value
                    ),
                ));
            }
            if let Some(ref mcs) = result.model_confidence_set {
                kv.push((
                    "Model Confidence Set".into(),
                    format!("{:?} (p={:.4})", mcs.included, mcs.mcs_p_value),
                ));
            }
            if let Some(ref conf) = result.selection_confidence {
                kv.push((
                    "Selection Confidence".into(),
                    format!("{} (DM p={:.4})", conf.verdict, conf.dm_p_value),
                ));
            }
            if let Some(ref scores) = result.metric_scores {
                for (name, ms) in scores {
                    kv.push((format!("Score: {}", name), format!("{}", ms)));
                }
            }
            if !kv.is_empty() {
                sections.push(ReportSection {
                    heading: "Model Selection".into(),
                    content: ReportContent::KeyValue(kv),
                });
            }
        }

        // 5. Ensemble
        if let Some(ref weights) = result.ensemble_weights {
            let rows: Vec<Vec<String>> = weights
                .iter()
                .map(|(name, w)| vec![name.clone(), format!("{:.4}", w)])
                .collect();
            sections.push(ReportSection {
                heading: "Ensemble".into(),
                content: ReportContent::Table {
                    headers: vec!["Model".into(), "Weight".into()],
                    rows,
                },
            });
        }

        // 6. Forecast
        {
            let values = result.forecast.primary();
            let rows: Vec<Vec<String>> = values
                .iter()
                .enumerate()
                .map(|(i, v)| vec![format!("h={}", i + 1), format!("{:.4}", v)])
                .collect();
            sections.push(ReportSection {
                heading: "Forecast".into(),
                content: ReportContent::Table {
                    headers: vec!["Step".into(), "Value".into()],
                    rows,
                },
            });
        }

        // 7. Horizon Analysis
        if let Some(ref ha) = result.horizon_analysis {
            let rows: Vec<Vec<String>> = ha
                .steps
                .iter()
                .map(|step| {
                    vec![
                        format!("h={}", step.horizon),
                        format!("{:.4}", step.rmse),
                        format!("{:.4}", step.mae),
                        format!("{:.4}", step.bias),
                    ]
                })
                .collect();
            sections.push(ReportSection {
                heading: "Horizon Analysis".into(),
                content: ReportContent::Table {
                    headers: vec!["Step".into(), "RMSE".into(), "MAE".into(), "Bias".into()],
                    rows,
                },
            });
        }

        // 8. Decision Log
        if !result.log.is_empty() {
            let rows: Vec<Vec<String>> = result
                .log
                .decisions()
                .iter()
                .map(|d| {
                    vec![
                        format!("{}", d.step),
                        format!("{}", d.category),
                        d.action.clone(),
                        format!("{}", d.outcome),
                        d.detail.clone().unwrap_or_default(),
                    ]
                })
                .collect();
            sections.push(ReportSection {
                heading: "Decision Log".into(),
                content: ReportContent::Table {
                    headers: vec![
                        "#".into(),
                        "Category".into(),
                        "Action".into(),
                        "Outcome".into(),
                        "Detail".into(),
                    ],
                    rows,
                },
            });
        }

        // 9. Execution Metadata
        if !result.model_metadata.is_empty() {
            let rows: Vec<Vec<String>> = result
                .model_metadata
                .iter()
                .map(|m| {
                    vec![
                        m.model_name.clone(),
                        m.fit_duration
                            .map(|d| format!("{:.3}s", d.as_secs_f64()))
                            .unwrap_or_else(|| "-".into()),
                        if m.converged {
                            "yes".into()
                        } else {
                            "no".into()
                        },
                        m.error_message.clone().unwrap_or_default(),
                    ]
                })
                .collect();
            sections.push(ReportSection {
                heading: "Execution Metadata".into(),
                content: ReportContent::Table {
                    headers: vec![
                        "Model".into(),
                        "Fit Time".into(),
                        "Converged".into(),
                        "Error".into(),
                    ],
                    rows,
                },
            });
        }

        PipelineReport {
            title: format!("Pipeline Report: {}", result.model_name),
            sections,
        }
    }
}

impl fmt::Display for PipelineReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.title)?;
        writeln!(f, "{}", "=".repeat(self.title.len()))?;
        writeln!(f)?;

        for section in &self.sections {
            writeln!(f, "--- {} ---", section.heading)?;
            match &section.content {
                ReportContent::Text(text) => {
                    writeln!(f, "{}", text)?;
                }
                ReportContent::KeyValue(pairs) => {
                    let max_key = pairs.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
                    for (k, v) in pairs {
                        writeln!(f, "  {:width$}  {}", k, v, width = max_key)?;
                    }
                }
                ReportContent::Table { headers, rows } => {
                    // Calculate column widths
                    let n_cols = headers.len();
                    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
                    for row in rows {
                        for (i, cell) in row.iter().enumerate() {
                            if i < n_cols {
                                widths[i] = widths[i].max(cell.len());
                            }
                        }
                    }

                    // Header
                    let header_line: String = headers
                        .iter()
                        .zip(widths.iter())
                        .map(|(h, w)| format!("  {:width$}", h, width = w))
                        .collect();
                    writeln!(f, "{}", header_line)?;
                    let sep: String = widths
                        .iter()
                        .map(|w| format!("  {}", "-".repeat(*w)))
                        .collect();
                    writeln!(f, "{}", sep)?;

                    // Rows
                    for row in rows {
                        let line: String = row
                            .iter()
                            .zip(widths.iter())
                            .map(|(cell, w)| format!("  {:width$}", cell, width = w))
                            .collect();
                        writeln!(f, "{}", line)?;
                    }
                }
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Forecast;
    use crate::orchestration::decision_log::{DecisionCategory, DecisionLog, DecisionOutcome};
    use crate::orchestration::metadata::ExecutionMetadata;
    use std::time::Duration as StdDuration;

    fn make_result() -> PipelineResult {
        let forecast = Forecast::from_values(vec![1.0, 2.0, 3.0]);
        let mut log = DecisionLog::new();
        log.record(
            DecisionCategory::ModelSelection,
            "Selected Naive",
            DecisionOutcome::Success,
        );
        PipelineResult {
            forecast,
            model_name: "Naive".into(),
            profile: None,
            log,
            model_metadata: vec![ExecutionMetadata::new("Naive")
                .with_fit(StdDuration::from_millis(10))
                .with_convergence(true)],
            horizon_analysis: None,
            selection_confidence: None,
            model_confidence_set: None,
            quality_floor: None,
            preprocess: None,
            ensemble_weights: None,
            metric_scores: None,
            trend_selection: None,
            seasonal_selection: None,
            structural_break_in_holdout: false,
            holdout_forecasts: None,
            holdout_actual: None,
        }
    }

    #[test]
    fn report_from_result() {
        let result = make_result();
        let report = PipelineReport::from_result(&result);
        assert!(report.title.contains("Naive"));
        assert!(!report.sections.is_empty());
    }

    #[test]
    fn report_display() {
        let result = make_result();
        let report = PipelineReport::from_result(&result);
        let text = format!("{}", report);
        assert!(text.contains("Pipeline Report"));
        assert!(text.contains("Summary"));
        assert!(text.contains("Forecast"));
        assert!(text.contains("Decision Log"));
    }

    #[test]
    fn report_with_ensemble_weights() {
        let mut result = make_result();
        result.ensemble_weights = Some(vec![("SES".into(), 0.6), ("Naive".into(), 0.4)]);
        let report = PipelineReport::from_result(&result);
        let text = format!("{}", report);
        assert!(text.contains("Ensemble"));
        assert!(text.contains("SES"));
    }

    #[test]
    fn report_structural_break_warning() {
        let mut result = make_result();
        result.structural_break_in_holdout = true;
        let report = PipelineReport::from_result(&result);
        let text = format!("{}", report);
        assert!(text.contains("Warnings"));
        assert!(text.contains("Structural break"));
    }

    #[test]
    fn report_no_warning_when_clean() {
        let result = make_result();
        let report = PipelineReport::from_result(&result);
        let text = format!("{}", report);
        assert!(!text.contains("Warnings"));
    }
}
