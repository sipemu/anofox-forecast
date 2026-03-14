//! Orchestration module example — data profiling, pipeline execution,
//! fallback chains, horizon analysis, and selection confidence.
//!
//! Run with: cargo run --example orchestration

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::baseline::{
    HistoricAverage, Naive, SeasonalNaive, SimpleMovingAverage,
};
use anofox_forecast::models::exponential::SimpleExponentialSmoothing;
use anofox_forecast::models::{Forecaster, ModelRegistry, ModelSpec};
use anofox_forecast::orchestration::prelude::*;
use anofox_forecast::orchestration::report::PipelineReport;
use anofox_forecast::orchestration::store::{InMemoryStore, PipelineStore, RecordKind};
use anofox_forecast::orchestration::tools;
use chrono::{Duration, TimeZone, Utc};

/// Build a sample time series with trend + seasonality.
fn make_series(n: usize) -> TimeSeries {
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect();

    let values: Vec<f64> = (0..n)
        .map(|i| {
            50.0 + 0.3 * i as f64                   // trend
                + 10.0 * (i as f64 * 2.0 * std::f64::consts::PI / 7.0).sin() // weekly cycle
                + 2.0 * (i as f64 * 0.1).cos() // noise-like variation
        })
        .collect();

    TimeSeries::univariate(timestamps, values).unwrap()
}

fn main() {
    println!("=== Orchestration Module Demo ===\n");

    let ts = make_series(120);

    // ───────────────────────────────────────────────────────────
    // 1. Data Profiling
    // ───────────────────────────────────────────────────────────
    println!("─── 1. Data Profiling ───\n");

    let profile = DataProfile::from_series(&ts);
    println!("{}\n", profile);

    // Access individual fields
    println!("  Stationary (ADF)?  {}", profile.adf_is_stationary);
    println!("  Trend direction:   {}", profile.trend_direction);
    println!("  ACF lag-1:         {:.4}", profile.acf_lag1);
    println!("  Quality score:     {:.2}\n", profile.quality_score);

    // ───────────────────────────────────────────────────────────
    // 2. Decision Log
    // ───────────────────────────────────────────────────────────
    println!("─── 2. Decision Log ───\n");

    let mut log = DecisionLog::new();
    log.record(
        DecisionCategory::DataProfiling,
        "Profiled 120 observations",
        DecisionOutcome::Success,
    );
    log.record_with_detail(
        DecisionCategory::ModelFitting,
        "Fitted SES",
        DecisionOutcome::Success,
        "MAE=4.32",
    );
    log.record_timed(
        DecisionCategory::CrossValidation,
        "CV fold 1",
        DecisionOutcome::Success,
        std::time::Duration::from_millis(42),
    );
    log.record(
        DecisionCategory::ModelFitting,
        "Fitted ARIMA(2,1,1)",
        DecisionOutcome::Failed,
    );
    println!("{}\n", log);

    // Query the log
    println!("  Failures: {}", log.failures().len());
    println!(
        "  Fitting decisions: {}\n",
        log.by_category(DecisionCategory::ModelFitting).len()
    );

    // ───────────────────────────────────────────────────────────
    // 3. Fallback Chain
    // ───────────────────────────────────────────────────────────
    println!("─── 3. Fallback Chain ───\n");

    let chain = FallbackChain::new()
        .add("Naive", || Box::new(Naive::new()))
        .add("SMA(5)", || Box::new(SimpleMovingAverage::new(5)))
        .add("HistoricAverage", || Box::new(HistoricAverage::new()));

    let result = chain.execute(&ts, 7).unwrap();
    println!("  Winner:    {}", result.model_name);
    println!("  Attempts:  {}", result.attempts);
    println!("  Failed:    {:?}", result.failed_models);
    println!("  Forecast:  {:?}\n", &result.forecast.primary()[..3]);

    // Fallback with logging
    let mut fb_log = DecisionLog::new();
    let _ = chain.execute_with_log(&ts, 7, &mut fb_log);
    println!("  Fallback log:\n{}\n", fb_log);

    // ───────────────────────────────────────────────────────────
    // 4. Execution Metadata
    // ───────────────────────────────────────────────────────────
    println!("─── 4. Execution Metadata ───\n");

    let timer = ExecutionTimer::start();
    let mut ses = SimpleExponentialSmoothing::new(0.3);
    ses.fit(&ts).unwrap();
    let fit_dur = timer.stop();

    let timer2 = ExecutionTimer::start();
    let _ = ses.predict(7).unwrap();
    let pred_dur = timer2.stop();

    let meta = ExecutionMetadata::new("SES(0.3)")
        .with_fit(fit_dur)
        .with_predict(pred_dur)
        .with_observations(ts.len())
        .with_horizon(7)
        .with_convergence(true);

    println!("{}\n", meta);

    // ───────────────────────────────────────────────────────────
    // 5. Selection Confidence
    // ───────────────────────────────────────────────────────────
    println!("─── 5. Selection Confidence ───\n");

    // 5a. Diebold-Mariano pairwise test
    let ses_fold_errors = vec![4.2, 4.5, 3.9, 4.1, 4.3, 4.0, 3.8, 4.4, 4.1, 3.9];
    let naive_fold_errors = vec![8.1, 7.9, 8.3, 8.0, 8.2, 7.8, 8.4, 8.1, 7.9, 8.0];

    let confidence =
        SelectionConfidence::compare("SES", ses_fold_errors, "Naive", naive_fold_errors);
    println!("  DM test: {}", confidence);
    println!("  Significant? {}\n", confidence.is_significant());

    // 5b. Model Confidence Set — which models are statistically best?
    let mcs = ModelConfidenceSet::from_cv_scores(
        vec![
            (
                "SES".into(),
                vec![4.2, 4.5, 3.9, 4.1, 4.3, 4.0, 3.8, 4.4, 4.1, 3.9],
            ),
            (
                "Naive".into(),
                vec![8.1, 7.9, 8.3, 8.0, 8.2, 7.8, 8.4, 8.1, 7.9, 8.0],
            ),
            (
                "SMA(5)".into(),
                vec![5.0, 5.2, 4.8, 5.1, 5.0, 4.9, 5.1, 5.0, 4.8, 5.2],
            ),
        ],
        0.10,
    );
    if let Some(ref mcs) = mcs {
        println!("  {}", mcs);
        println!("  Single winner? {}\n", mcs.has_single_winner());
    }

    // 5c. Quality Floor (SPA test) — does any model beat the benchmark?
    let benchmark_losses = vec![8.0; 50];
    let candidate_losses = vec![vec![3.0; 50], vec![5.0; 50]];
    let qf = QualityFloor::test("Naive", &benchmark_losses, &candidate_losses);
    if let Some(ref qf) = qf {
        println!("  {}\n", qf);
    }

    // ───────────────────────────────────────────────────────────
    // 6. Horizon Analysis
    // ───────────────────────────────────────────────────────────
    println!("─── 6. Horizon Analysis ───\n");

    // Simulate 3 CV folds, each with 7-step ahead actuals/forecasts
    let actuals: Vec<Vec<f64>> = vec![
        vec![50.0, 52.0, 48.0, 55.0, 51.0, 49.0, 53.0],
        vec![51.0, 53.0, 47.0, 56.0, 50.0, 48.0, 54.0],
        vec![49.0, 51.0, 49.0, 54.0, 52.0, 50.0, 52.0],
    ];
    let forecasts: Vec<Vec<f64>> = vec![
        vec![50.5, 51.0, 50.0, 52.0, 51.5, 50.5, 51.0],
        vec![51.5, 52.0, 49.0, 53.0, 51.0, 49.5, 52.0],
        vec![49.5, 50.5, 48.5, 53.5, 51.5, 50.0, 51.5],
    ];

    let a_refs: Vec<&[f64]> = actuals.iter().map(|v| v.as_slice()).collect();
    let f_refs: Vec<&[f64]> = forecasts.iter().map(|v| v.as_slice()).collect();

    let horizon = HorizonAnalysis::from_folds("SES", &a_refs, &f_refs);
    println!("{}", horizon);

    if let Some(hardest) = horizon.hardest_horizon() {
        println!(
            "  Hardest step: h={} (RMSE={:.4})",
            hardest.horizon, hardest.rmse
        );
    }
    if let Some(easiest) = horizon.easiest_horizon() {
        println!(
            "  Easiest step: h={} (RMSE={:.4})",
            easiest.horizon, easiest.rmse
        );
    }
    if let Some(growth) = horizon.error_growth_rate() {
        println!("  Error growth rate: {:.1}%\n", growth * 100.0);
    }

    // ───────────────────────────────────────────────────────────
    // 7. Pipeline (end-to-end)
    // ───────────────────────────────────────────────────────────
    println!("─── 7. Pipeline (end-to-end) ───\n");

    let mut registry = ModelRegistry::new();
    registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
    registry.register(ModelSpec::new(
        "SMA(5)",
        || Box::new(SimpleMovingAverage::new(5)),
        false,
    ));
    registry.register(ModelSpec::new(
        "HistoricAverage",
        || Box::new(HistoricAverage::new()),
        false,
    ));
    registry.register(ModelSpec::with_period(
        "SeasonalNaive(7)",
        |p| Box::new(SeasonalNaive::new(p)),
        7,
        true,
    ));
    registry.register(ModelSpec::new(
        "SES(0.3)",
        || Box::new(SimpleExponentialSmoothing::new(0.3)),
        true,
    ));

    let result = PipelineBuilder::new()
        .profile()
        .preprocess(PreprocessMode::Auto)
        .metric(MetricStrategy::Auto)
        .ensemble(EnsembleMode::Auto)
        .registry(registry)
        .select_models(3)
        .with_fallback()
        .non_negative()
        .build()
        .execute(&ts, 7)
        .unwrap();

    println!("{}", result);

    println!("  Forecast (first 7 steps):");
    for (i, v) in result.forecast.primary().iter().enumerate() {
        println!("    h={}: {:.4}", i + 1, v);
    }

    if let Some(ref qf) = result.quality_floor {
        println!("\n  {}", qf);
    }
    if let Some(ref mcs) = result.model_confidence_set {
        println!("  {}", mcs);
    }
    if let Some(ref conf) = result.selection_confidence {
        println!("  DM: {}", conf);
    }

    // ───────────────────────────────────────────────────────────
    // 8. Pipeline from saved config (replay)
    // ───────────────────────────────────────────────────────────
    println!("\n─── 8. Pipeline Replay from Config ───\n");

    let config = PipelineConfig {
        profile: true,
        select_top_k: 0,
        cv_folds: 0,
        horizon: 7,
        holdout: 7,
        use_fallback: true,
        interval_level: 0.0,
        non_negative: true,
        horizon_analysis: false,
        seasonal_period: 0,
        preprocess: PreprocessMode::default(),
        metric_strategy: MetricStrategy::default(),
        ensemble_mode: EnsembleMode::default(),
        postprocess_coverage: 0.0,
    };
    println!("  Config: {:?}\n", config);

    // Replay on new data with a fresh registry
    let new_ts = make_series(90);
    let mut reg2 = ModelRegistry::new();
    reg2.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
    reg2.register(ModelSpec::new(
        "SMA(5)",
        || Box::new(SimpleMovingAverage::new(5)),
        false,
    ));

    let replay = Pipeline::from_config(config)
        .with_registry(reg2)
        .execute(&new_ts, 7)
        .unwrap();

    println!("  Replay model: {}", replay.model_name);
    println!("  Replay forecast: {:?}", &replay.forecast.primary()[..3]);

    // ───────────────────────────────────────────────────────────
    // 9. Pipeline Report
    // ───────────────────────────────────────────────────────────
    println!("\n─── 9. Pipeline Report ───\n");

    let report = PipelineReport::from_result(&result);
    println!("{}", report);

    // ───────────────────────────────────────────────────────────
    // 10. Multi-Metric Selection
    // ───────────────────────────────────────────────────────────
    println!("─── 10. Multi-Metric Selection ───\n");

    // Auto strategy selects metrics based on data characteristics
    let auto_strat = MetricStrategy::Auto;
    println!(
        "  Auto (non-intermittent, non-negative): {}",
        auto_strat.description(false, false)
    );
    println!(
        "  Auto (intermittent):                   {}",
        auto_strat.description(true, false)
    );
    println!(
        "  Auto (general, has negatives):          {}",
        auto_strat.description(false, true)
    );

    // Custom composite strategy
    let custom = MetricStrategy::Composite(vec![
        (Metric::MAE, 0.4),
        (Metric::RMSE, 0.3),
        (Metric::MDA, 0.3),
    ]);
    let actual = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let predicted = vec![12.0, 18.0, 33.0, 38.0, 52.0];
    let scores = custom.score(&actual, &predicted, false, false);
    println!("  Custom score: {}\n", scores);

    // ───────────────────────────────────────────────────────────
    // 11. Structured Tools (MCP-ready)
    // ───────────────────────────────────────────────────────────
    println!("─── 11. Structured Tools ───\n");

    // Profile data tool
    let prof_output = tools::profile_data(tools::ProfileDataInput { series: &ts });
    println!(
        "  profile_data → {} observations, quality={:.2}",
        prof_output.profile.n_observations, prof_output.profile.quality_score
    );

    // Select models tool
    let sel_output = tools::select_models(tools::SelectModelsInput {
        profile: &prof_output.profile,
        available_models: &[],
    });
    println!("  select_models → {:?}", sel_output.recommended);
    for reason in &sel_output.reasoning {
        println!("    - {}", reason);
    }

    // ───────────────────────────────────────────────────────────
    // 12. Abstract Storage (InMemoryStore)
    // ───────────────────────────────────────────────────────────
    println!("\n─── 12. Abstract Storage ───\n");

    let store = InMemoryStore::new();
    let record = anofox_forecast::orchestration::store::PipelineRecord {
        id: "run-001".into(),
        timestamp: chrono::Utc::now(),
        kind: RecordKind::Result,
        fields: Value::map_from(vec![
            ("model", Value::String("SES".into())),
            ("horizon", Value::Int(7)),
            ("score", Value::Float(4.32)),
        ]),
    };
    store.save(&record).unwrap();

    let ids = store.list(None).unwrap();
    println!("  Stored records: {:?}", ids);

    let loaded = store.load("run-001").unwrap();
    if let Some(rec) = loaded {
        println!(
            "  Loaded: kind={}, model={:?}",
            rec.kind,
            rec.fields.get("model")
        );
    }

    println!("\n=== Done ===");
}
