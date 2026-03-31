use anofox_forecast::changepoint::{CostFunction, Pelt};

#[test]
fn auto_detect_level_shifts() {
    // Three segments: 0, 10, 5
    let mut series = Vec::new();
    for _ in 0..100 {
        series.push(0.0 + ((series.len() * 7 + 3) % 11) as f64 * 0.2 - 1.0);
    }
    for _ in 0..100 {
        series.push(10.0 + ((series.len() * 7 + 3) % 11) as f64 * 0.2 - 1.0);
    }
    for _ in 0..100 {
        series.push(5.0 + ((series.len() * 7 + 3) % 11) as f64 * 0.2 - 1.0);
    }

    let result = Pelt::new(CostFunction::L2)
        .min_size(10)
        .auto_detect(&series);

    println!(
        "Auto-detected {} changepoints at {:?}",
        result.result.n_changepoints, result.result.changepoints
    );
    println!("Selected penalty: {:.2}", result.penalty);
    println!(
        "CROPS explored {} distinct segmentations",
        result.crops.len()
    );
    for (pen, r) in &result.crops {
        println!(
            "  penalty={:.2} → {} changepoints (cost={:.1})",
            pen, r.n_changepoints, r.cost
        );
    }

    // Should detect 2 changepoints near indices 100 and 200
    assert!(
        result.result.n_changepoints >= 2,
        "Expected at least 2 changepoints, got {}",
        result.result.n_changepoints
    );

    // Changepoints should be near 100 and 200
    let cps = &result.result.changepoints;
    assert!(
        cps.iter().any(|&cp| (cp as i64 - 100).abs() < 10),
        "No changepoint near 100: {:?}",
        cps
    );
    assert!(
        cps.iter().any(|&cp| (cp as i64 - 200).abs() < 10),
        "No changepoint near 200: {:?}",
        cps
    );
}

#[test]
fn auto_detect_no_changepoint() {
    // Constant series with noise — should detect 0 or very few changepoints
    let series: Vec<f64> = (0..200)
        .map(|i| 5.0 + ((i * 7 + 3) % 11) as f64 * 0.2 - 1.0)
        .collect();

    let result = Pelt::new(CostFunction::L2)
        .min_size(10)
        .auto_detect(&series);

    println!(
        "Constant series: {} changepoints, penalty={:.2}",
        result.result.n_changepoints, result.penalty
    );
    assert!(
        result.result.n_changepoints <= 2,
        "Too many changepoints on constant series: {}",
        result.result.n_changepoints
    );
}

#[test]
fn crops_returns_distinct_segmentations() {
    let mut series = vec![0.0; 100];
    series.extend(vec![10.0; 100]);

    let crops = Pelt::new(CostFunction::L2)
        .min_size(5)
        .crops(&series, None, None);

    println!("CROPS: {} distinct segmentations", crops.len());
    for (pen, r) in &crops {
        println!("  pen={:.2}: {} CPs", pen, r.n_changepoints);
    }

    // Should have at least 2 distinct results (0 CPs and 1+ CPs)
    assert!(
        crops.len() >= 2,
        "CROPS should find at least 2 distinct segmentations"
    );

    // Penalties should be monotonically increasing
    for i in 1..crops.len() {
        assert!(crops[i].0 >= crops[i - 1].0, "Penalties should be sorted");
    }

    // n_changepoints should be monotonically decreasing (or equal)
    for i in 1..crops.len() {
        assert!(
            crops[i].1.n_changepoints <= crops[i - 1].1.n_changepoints,
            "More penalty should mean fewer changepoints"
        );
    }
}
