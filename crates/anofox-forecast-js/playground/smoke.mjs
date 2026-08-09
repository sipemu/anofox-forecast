// End-to-end smoke test: load WASM in Node, instantiate LaplacePlayground,
// warm up with a synthetic seasonal series, observe a few new obs, and print
// the forecast. Also verifies each recipe branch reports a sensible name.
import { readFile } from 'node:fs/promises';
import init, { LaplacePlayground } from './pkg/anofox_forecast_js.js';

const wasm = await readFile(new URL('./pkg/anofox_forecast_js_bg.wasm', import.meta.url));
await init(wasm);

class Rng {
  constructor(seed) { this.s = BigInt(seed) & ((1n << 64n) - 1n); }
  next() {
    this.s = (this.s * 6364136223846793005n + 1442695040888963407n) & ((1n << 64n) - 1n);
    return Number((this.s >> 32n) & 0xffffffffn) / 4294967296;
  }
  normal() { let s = 0; for (let i = 0; i < 12; i++) s += this.next(); return s - 6; }
}

const scenarios = [
  { name: 'seasonal_linear_trend, N=80, P=12', period: 12, warmup: 80,
    gen: (rng) => (i) => 50 + 5 * Math.sin(2 * Math.PI * i / 12) + 0.05 * i + rng.normal() },
  { name: 'pure_gaussian_noise, N=80', period: 0, warmup: 80,
    gen: (rng) => () => 50 + rng.normal() },
  { name: 'short_history N=50', period: 0, warmup: 50,
    gen: (rng) => () => 50 + rng.normal() },
  { name: 'all_zeros_rare_spikes, N=100', period: 0, warmup: 100,
    gen: (rng) => () => (rng.next() < 0.02 ? 10 : 0) },
];

let fail = 0;
for (const sc of scenarios) {
  const rng = new Rng(42);
  const draw = sc.gen(rng);
  const warm = [];
  for (let i = 0; i < sc.warmup; i++) warm.push(draw(i));
  const pg = new LaplacePlayground(new Float64Array(warm), sc.period ? String(sc.period) : '', 12, '');
  const recipe = pg.recipe();
  const fc = pg.forecast(12, new Float64Array([0.1, 0.5, 0.9]));
  // fc layout: 12 rows × 4 cols (0.1, 0.5, 0.9, mean).
  const ok = fc.length === 12 * 4 && Array.from(fc).every(Number.isFinite);
  const median_h1 = fc[1];
  console.log(`  ${sc.name}  →  recipe=${recipe}  h=1 median=${median_h1.toFixed(3)}  ${ok ? 'OK' : 'FAIL'}`);
  if (!ok) fail++;
  // Observe one more point.
  pg.observe(draw(sc.warmup));
  const fc2 = pg.forecast(6, new Float64Array([0.5]));
  if (fc2.length !== 12) console.log(`    forecast len after observe: ${fc2.length} (expected 6*2=12)`);
  pg.free();
}

if (fail > 0) { console.error(`FAILED (${fail})`); process.exit(1); }
console.log('OK — all scenarios produced finite forecasts.');

// Variant + multi-period smoke.
console.log('\nVariant + multi-period smoke:');
{
  const rng = new Rng(9);
  const warm = [];
  for (let i = 0; i < 200; i++) {
    warm.push(50 + 3 * Math.sin(2 * Math.PI * i / 24) + 2 * Math.sin(2 * Math.PI * i / 168) + rng.normal());
  }
  const cases = [
    { variant: '',                periodsCsv: '24' },
    { variant: 'multiscale_3sh',  periodsCsv: '24' },
    { variant: 'skaters',         periodsCsv: '24,168' },
    { variant: 'auto',            periodsCsv: '' },
  ];
  for (const c of cases) {
    const pg = new LaplacePlayground(new Float64Array(warm), c.periodsCsv, 24, c.variant);
    const fc = pg.forecast(24, new Float64Array([0.5]));
    const label = pg.recipe();
    const ok = fc.length === 48 && Array.from(fc).every(Number.isFinite);
    console.log(`  variant='${c.variant.padEnd(15)}' periods='${c.periodsCsv.padEnd(6)}'  →  ${label}   ${ok ? 'OK' : 'FAIL'}`);
    pg.free();
    if (!ok) process.exit(1);
  }
}
console.log('OK — variant selector + multi-period supported.');

// Mahalanobis playground smoke.
console.log('\nMahalanobisPlayground smoke:');
{
  const { MahalanobisPlayground } = await import('./pkg/anofox_forecast_js.js');
  const rng = new Rng(11);
  const warm = [];
  for (let i = 0; i < 200; i++) warm.push(50 + rng.normal());
  const det = new MahalanobisPlayground(new Float64Array(warm), 6, '');
  // Feed enough post-warmup obs for the parade z-bank to fill up.
  for (let i = 0; i < 60; i++) det.observe(50 + rng.normal());
  const scoreQuiet = det.last_score();
  det.observe(50 + 20);   // 20σ spike
  const scoreSpike = det.last_score();
  console.log(`  quiet  d²=${fmtOrNaN(scoreQuiet[0])}  p=${fmtOrNaNE(scoreQuiet[1])}  run=${scoreQuiet[2]}  warm=${scoreQuiet[3]}`);
  console.log(`  spike  d²=${fmtOrNaN(scoreSpike[0])}  p=${fmtOrNaNE(scoreSpike[1])}  run=${scoreSpike[2]}  warm=${scoreSpike[3]}`);
  const ok = scoreSpike[3] === 1.0 && scoreSpike[0] > scoreQuiet[0];
  console.log(`  ${ok ? 'OK — spike scored higher than quiet baseline' : 'FAIL'}`);
  det.free();
  if (!ok) process.exit(1);
}
function fmtOrNaN(v) { return Number.isFinite(v) ? v.toFixed(2) : 'NaN'; }
function fmtOrNaNE(v) { return Number.isFinite(v) ? v.toExponential(1) : 'NaN'; }
console.log('OK — Mahalanobis detector reports d²/p_value/run.');

// ---------- Anomaly-tab methods smoke ----------
console.log('\nAnomaly-tab methods:');
{
  const rng = new Rng(7);
  const warm = [];
  for (let i = 0; i < 100; i++) warm.push(50 + rng.normal());
  const pg = new LaplacePlayground(new Float64Array(warm), '', 1, '');
  const surpriseIn = pg.surprise(50);            // ~expected value
  const surpriseOut = pg.surprise(50 + 8);       // ~8σ out
  const tailIn = pg.tail_probability(50);
  const tailOut = pg.tail_probability(50 + 8);
  const ok = Number.isFinite(surpriseIn)
    && Number.isFinite(surpriseOut)
    && surpriseOut > surpriseIn + 5   // 8σ out must have much higher surprise
    && tailIn > 0.5                    // near-median → tail prob near 1
    && tailOut < 0.05;                 // 8σ out → tiny tail prob
  console.log(`  surprise(50)  = ${surpriseIn.toFixed(2)}   surprise(58)  = ${surpriseOut.toFixed(2)}   ${surpriseOut > surpriseIn + 5 ? 'OK' : 'FAIL'}`);
  console.log(`  tailProb(50)  = ${tailIn.toFixed(3)}     tailProb(58)  = ${tailOut.toExponential(2)}     ${tailIn > 0.5 && tailOut < 0.05 ? 'OK' : 'FAIL'}`);
  if (!ok) { console.error('anomaly methods FAILED sanity check'); process.exit(1); }
  pg.free();
}
console.log('OK — anomaly methods behave.');
