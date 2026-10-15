//! Faithful port of R `stats::arima`'s estimation path as driven by
//! `forecast::auto.arima`'s `myarima` (D-06, 11-12).
//!
//! 11-11 proved the exact Gaussian likelihood formula
//! (`exact_likelihood::exact_concentrated_loglik`) matches R to within
//! 0.002 AICc at R's own coefficients, but AutoARIMA still diverged
//! from R's picks because the crate *optimised* that likelihood
//! differently (box-bounded Nelder-Mead/L-BFGS from ad-hoc starts) --
//! which changes which candidates converge, to where, and which ones R
//! rejects outright. This module reproduces R's optimiser trajectory
//! step by step:
//!
//! - **CSS stage** (`armaCSS`): `0.5 * log(ssq / nu)` with R's
//!   `ARIMA_CSS` recursion conditioned on `ncond = p + P*m` observations
//!   of the differenced series, minimised by `optim(method = "BFGS")`
//!   from `init0` (all ARMA coefficients 0, the mean/drift coefficient
//!   at its `lm` estimate) with `parscale` (1 for ARMA, `10 * se` for the
//!   regression coefficient). The CSS result is used as the ML start
//!   only when `optim` converged (`convergence == 0`), exactly like R.
//! - R's `arCheck` on the CSS start (strict `Mod(polyroot) > 1`), which
//!   raises "non-stationary AR part from CSS" in R (`ic = Inf`).
//! - **ML stage** (`armafn`) with `transform.pars = TRUE`: AR and
//!   seasonal-AR blocks re-parameterised through R's `partrans`
//!   (`tanh` + Durbin-Levinson), MA blocks started from `maInvert`ed CSS
//!   values, minimised by the same BFGS port; final MA blocks
//!   `maInvert`ed and, if that changed anything, re-evaluated (R's
//!   `maxit = 0` re-run).
//! - `optim`'s `vmmin` (BFGS) and `fmingr` (central differences with
//!   `ndeps = 1e-3` in `parscale` units), `reltol = sqrt(.Machine$double.eps)`,
//!   `maxit = 100`, and `optimhess` for the Hessian.
//! - `forecast:::checkarima`: `var.coef = A' (H * n.used)^-1 A`
//!   (`A = ARIMA_Gradtrans`); a NaN standard error, or a singular
//!   Hessian (`solve()` error), makes R score the candidate `Inf`.
//! - `myarima`'s information criteria: CSS-ML uses `aic = -2*loglik +
//!   2*npar`; CSS (`approximation = TRUE`) uses `nstar*log(sigma2) +
//!   2*npar` (R's `offset` is a per-series constant and is omitted -- it
//!   never changes a ranking), `nstar = n - d - D*m`,
//!   `npar = #coef + 1`, AICc/BIC from `aic`.
//!
//! The exact likelihood itself is evaluated on the differenced series
//! through the innovations algorithm (`exact_concentrated_loglik`),
//! which is what R's diffuse-prior Kalman filter computes on the
//! undifferenced series up to its `kappa = 1e6` approximation.

use super::exact_likelihood::{expand_seasonal_ar, expand_seasonal_ma, seasonal_roots_outside};

/// R `optim()` BFGS defaults.
const NDEPS: f64 = 1e-3;
const MAXIT: usize = 100;
/// `sqrt(.Machine$double.eps)`.
const RELTOL: f64 = 1.490_116_119_384_765_6e-8;

/// Which `stats::arima` `method` to emulate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RMethod {
    /// `method = "CSS"` (`auto.arima`'s `approximation = TRUE` search).
    Css,
    /// `method = "CSS-ML"` (exact likelihood, R's default).
    CssMl,
}

/// A candidate fitted the way `myarima` fits it, with R's information
/// criteria. `None` from [`r_arima_fit`] is R's `ic = Inf`.
#[derive(Debug, Clone)]
#[allow(dead_code)] // loglik/sigma2/aic kept for diagnostics and parity tests
pub(crate) struct RArimaFit {
    pub loglik: f64,
    pub sigma2: f64,
    pub aic: f64,
    pub aicc: f64,
    pub bic: f64,
    pub ar: Vec<f64>,
    pub ma: Vec<f64>,
    pub sar: Vec<f64>,
    pub sma: Vec<f64>,
    /// Mean (d+D = 0) or drift (d+D = 1) coefficient, 0 when absent.
    pub constant: f64,
}

// ---------------------------------------------------------------------------
// Minimal complex arithmetic + polynomial roots (R `polyroot` stand-in)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

/// Roots of `c[0] + c[1] z + ... + c[k] z^k` (R `polyroot`'s ascending
/// convention, `c[k] != 0`), via Durand-Kerner iteration. Orders here
/// are small (non-seasonal <= 5, seasonal <= 2), where this converges
/// to machine precision.
fn poly_roots(c: &[f64]) -> Option<Vec<C64>> {
    let k = c.len().checked_sub(1)?;
    if k == 0 {
        return Some(vec![]);
    }
    let lead = c[k];
    if lead == 0.0 || !lead.is_finite() {
        return None;
    }
    // Monic, descending: z^k + a[1] z^(k-1) + ... + a[k].
    let a: Vec<f64> = (0..=k).map(|i| c[k - i] / lead).collect();
    let eval = |z: C64| -> C64 {
        let mut acc = C64::new(1.0, 0.0);
        for &ai in &a[1..] {
            acc = acc.mul(z).add(C64::new(ai, 0.0));
        }
        acc
    };
    let radius = 1.0 + a[1..].iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut z: Vec<C64> = (0..k)
        .map(|i| {
            let ang = 2.0 * std::f64::consts::PI * i as f64 / k as f64 + 0.4;
            C64::new(radius * 0.5 * ang.cos(), radius * 0.5 * ang.sin())
        })
        .collect();
    for _ in 0..500 {
        let mut max_step = 0.0_f64;
        for i in 0..k {
            let mut denom = C64::new(1.0, 0.0);
            for j in 0..k {
                if i != j {
                    denom = denom.mul(z[i].sub(z[j]));
                }
            }
            if denom.abs() == 0.0 {
                denom = C64::new(1e-12, 0.0);
            }
            let step = eval(z[i]).div(denom);
            z[i] = z[i].sub(step);
            max_step = max_step.max(step.abs() / z[i].abs().max(1e-300));
        }
        if max_step < 1e-15 {
            break;
        }
    }
    if z.iter().any(|r| !r.re.is_finite() || !r.im.is_finite()) {
        return None;
    }
    Some(z)
}

/// R `arima`'s local `arCheck`: every root of `1 - sum ar_i z^i`
/// (trailing exact zeros dropped) has modulus > 1.
fn ar_check(ar: &[f64]) -> bool {
    let mut poly = vec![1.0];
    poly.extend(ar.iter().map(|&v| -v));
    while poly.len() > 1 && *poly.last().unwrap() == 0.0 {
        poly.pop();
    }
    if poly.len() == 1 {
        return true;
    }
    match poly_roots(&poly) {
        Some(roots) => roots.iter().all(|r| r.abs() > 1.0),
        None => false,
    }
}

/// R `arima`'s local `maInvert`: reflect MA roots inside the unit
/// circle to their reciprocals.
fn ma_invert(ma: &[f64]) -> Vec<f64> {
    let q = ma.len();
    let mut poly = vec![1.0];
    poly.extend_from_slice(ma);
    while poly.len() > 1 && *poly.last().unwrap() == 0.0 {
        poly.pop();
    }
    let q0 = poly.len() - 1;
    if q0 == 0 {
        return ma.to_vec();
    }
    let Some(mut roots) = poly_roots(&poly) else {
        return ma.to_vec();
    };
    let inside: Vec<bool> = roots.iter().map(|r| r.abs() < 1.0).collect();
    if !inside.iter().any(|&b| b) {
        return ma.to_vec();
    }
    if q0 == 1 {
        let mut out = vec![0.0; q];
        out[0] = 1.0 / ma[0];
        return out;
    }
    for (r, &ins) in roots.iter_mut().zip(&inside) {
        if ins {
            *r = C64::new(1.0, 0.0).div(*r);
        }
    }
    // x <- 1; for (r in roots) x <- c(x, 0) - c(0, x)/r
    let mut x = vec![C64::new(1.0, 0.0)];
    for r in roots {
        let mut next = vec![C64::new(0.0, 0.0); x.len() + 1];
        for (i, &xi) in x.iter().enumerate() {
            next[i] = next[i].add(xi);
            next[i + 1] = next[i + 1].sub(xi.div(r));
        }
        x = next;
    }
    let mut out: Vec<f64> = x[1..].iter().map(|c| c.re).collect();
    out.resize(q, 0.0);
    out
}

/// R's C `partrans`: unconstrained -> PACF (`tanh`) -> AR coefficients
/// (Durbin-Levinson).
fn partrans(raw: &[f64]) -> Vec<f64> {
    let p = raw.len();
    let mut new: Vec<f64> = raw.iter().map(|v| v.tanh()).collect();
    let mut work = new.clone();
    for j in 1..p {
        let a = new[j];
        for k in 0..j {
            work[k] -= a * new[j - k - 1];
        }
        new[..j].copy_from_slice(&work[..j]);
    }
    new
}

/// R's C `invpartrans`: AR coefficients -> PACF -> `atanh`.
fn invpartrans(phi: &[f64]) -> Vec<f64> {
    let p = phi.len();
    let mut new = phi.to_vec();
    let mut work = phi.to_vec();
    for j in (1..p).rev() {
        let a = new[j];
        for k in 0..j {
            work[k] = (new[k] + a * new[j - k - 1]) / (1.0 - a * a);
        }
        new[..j].copy_from_slice(&work[..j]);
    }
    new.iter().map(|v| v.atanh()).collect()
}

// ---------------------------------------------------------------------------
// optim(method = "BFGS") port: vmmin + fmingr + optimhess
// ---------------------------------------------------------------------------

struct Scaled<'a, F: Fn(&[f64]) -> f64> {
    f: &'a F,
    parscale: &'a [f64],
}

impl<F: Fn(&[f64]) -> f64> Scaled<'_, F> {
    fn value(&self, p: &[f64]) -> f64 {
        let x: Vec<f64> = p.iter().zip(self.parscale).map(|(a, s)| a * s).collect();
        (self.f)(&x)
    }

    /// `optim`'s numerical `fmingr`: central differences, step
    /// `ndeps` in scaled units. A non-finite difference is an error in
    /// R (`None` here).
    fn grad(&self, p: &[f64]) -> Option<Vec<f64>> {
        let n = p.len();
        let mut x: Vec<f64> = p.iter().zip(self.parscale).map(|(a, s)| a * s).collect();
        let mut df = vec![0.0; n];
        for i in 0..n {
            x[i] = (p[i] + NDEPS) * self.parscale[i];
            let v1 = (self.f)(&x);
            x[i] = (p[i] - NDEPS) * self.parscale[i];
            let v2 = (self.f)(&x);
            df[i] = (v1 - v2) / (2.0 * NDEPS);
            if !df[i].is_finite() {
                return None;
            }
            x[i] = p[i] * self.parscale[i];
        }
        Some(df)
    }
}

struct OptimResult {
    par: Vec<f64>,
    value: f64,
    convergence: i32,
}

/// R `src/appl/optim.c::vmmin` (variable-metric BFGS), in scaled
/// coordinates. `None` reproduces R's errors (non-finite initial value
/// or non-finite finite-difference gradient).
fn vmmin<F: Fn(&[f64]) -> f64>(f: &F, init: &[f64], parscale: &[f64]) -> Option<OptimResult> {
    const STEPREDN: f64 = 0.2;
    const ACCTOL: f64 = 0.0001;
    const RELTEST: f64 = 10.0;
    let sc = Scaled { f, parscale };
    let n = init.len();
    let mut b: Vec<f64> = init.iter().zip(parscale).map(|(a, s)| a / s).collect();
    let unscale = |b: &[f64]| -> Vec<f64> { b.iter().zip(parscale).map(|(a, s)| a * s).collect() };

    let mut fval = sc.value(&b);
    if !fval.is_finite() {
        return None;
    }
    let mut fmin = fval;
    let mut gradcount = 1usize;
    let mut g = sc.grad(&b)?;
    let mut iter = 1usize;
    let mut ilast = gradcount;
    let mut bmat = vec![vec![0.0; n]; n];
    let mut t = vec![0.0; n];
    let mut x = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut count;

    loop {
        if ilast == gradcount {
            for (i, row) in bmat.iter_mut().enumerate() {
                row.iter_mut().take(i).for_each(|v| *v = 0.0);
                row[i] = 1.0;
            }
        }
        x.copy_from_slice(&b);
        c.copy_from_slice(&g);
        let mut gradproj = 0.0;
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..=i {
                s -= bmat[i][j] * g[j];
            }
            for j in (i + 1)..n {
                s -= bmat[j][i] * g[j];
            }
            t[i] = s;
            gradproj += s * g[i];
        }

        if gradproj < 0.0 {
            let mut steplength = 1.0;
            let mut accpoint = false;
            loop {
                count = 0;
                for i in 0..n {
                    b[i] = x[i] + steplength * t[i];
                    if RELTEST + x[i] == RELTEST + b[i] {
                        count += 1;
                    }
                }
                if count < n {
                    fval = sc.value(&b);
                    accpoint = fval.is_finite() && fval <= fmin + gradproj * steplength * ACCTOL;
                    if !accpoint {
                        steplength *= STEPREDN;
                    }
                }
                if count == n || accpoint {
                    break;
                }
            }
            let enough =
                fval > f64::NEG_INFINITY && (fval - fmin).abs() > RELTOL * (fmin.abs() + RELTOL);
            if !enough {
                count = n;
                fmin = fval;
            }
            if count < n {
                fmin = fval;
                g = sc.grad(&b)?;
                gradcount += 1;
                iter += 1;
                let mut d1 = 0.0;
                for i in 0..n {
                    t[i] *= steplength;
                    c[i] = g[i] - c[i];
                    d1 += t[i] * c[i];
                }
                if d1 > 0.0 {
                    let mut d2 = 0.0;
                    for i in 0..n {
                        let mut s = 0.0;
                        for j in 0..=i {
                            s += bmat[i][j] * c[j];
                        }
                        for j in (i + 1)..n {
                            s += bmat[j][i] * c[j];
                        }
                        x[i] = s;
                        d2 += s * c[i];
                    }
                    d2 = 1.0 + d2 / d1;
                    for i in 0..n {
                        for j in 0..=i {
                            bmat[i][j] += (d2 * t[i] * t[j] - x[i] * t[j] - t[i] * x[j]) / d1;
                        }
                    }
                } else {
                    ilast = gradcount;
                }
            } else if ilast < gradcount {
                count = 0;
                ilast = gradcount;
            }
        } else {
            count = 0;
            if ilast == gradcount {
                count = n;
            } else {
                ilast = gradcount;
            }
        }
        if iter >= MAXIT {
            break;
        }
        if gradcount - ilast > 2 * n {
            ilast = gradcount;
        }
        if count == n && ilast == gradcount {
            break;
        }
    }
    // vmmin leaves `b` at the last *tried* point when the final line
    // search failed (count == n restores nothing): R returns `b`/`Fmin`
    // as-is, which is what we mirror.
    Some(OptimResult {
        par: unscale(&b),
        value: fmin,
        convergence: if iter < MAXIT { 0 } else { 1 },
    })
}

/// R `optimhess` at `par` (unscaled), symmetrised, in original units.
fn optimhess<F: Fn(&[f64]) -> f64>(f: &F, par: &[f64], parscale: &[f64]) -> Option<Vec<Vec<f64>>> {
    let sc = Scaled { f, parscale };
    let n = par.len();
    let mut dpar: Vec<f64> = par.iter().zip(parscale).map(|(a, s)| a / s).collect();
    let mut h = vec![vec![0.0; n]; n];
    for i in 0..n {
        let eps = NDEPS / parscale[i];
        dpar[i] += eps;
        let df1 = sc.grad(&dpar)?;
        dpar[i] -= 2.0 * eps;
        let df2 = sc.grad(&dpar)?;
        for j in 0..n {
            h[i][j] = (df1[j] - df2[j]) / (2.0 * eps * parscale[i] * parscale[j]);
        }
        dpar[i] += eps;
    }
    for i in 0..n {
        for j in 0..i {
            let tmp = 0.5 * (h[i][j] + h[j][i]);
            h[i][j] = tmp;
            h[j][i] = tmp;
        }
    }
    Some(h)
}

/// `solve(a, b)` with R's `La_solve` singularity rule (error when the
/// reciprocal 1-norm condition number is below `.Machine$double.eps`).
fn r_solve(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    // Gauss-Jordan inverse with partial pivoting.
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut inv: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    for col in 0..n {
        let piv = (col..n).max_by(|&r1, &r2| m[r1][col].abs().total_cmp(&m[r2][col].abs()))?;
        if m[piv][col] == 0.0 || !m[piv][col].is_finite() {
            return None;
        }
        m.swap(col, piv);
        inv.swap(col, piv);
        let d = m[col][col];
        for j in 0..n {
            m[col][j] /= d;
            inv[col][j] /= d;
        }
        for r in 0..n {
            if r != col {
                let fct = m[r][col];
                if fct != 0.0 {
                    for j in 0..n {
                        m[r][j] -= fct * m[col][j];
                        inv[r][j] -= fct * inv[col][j];
                    }
                }
            }
        }
    }
    let norm1 = |mat: &[Vec<f64>]| -> f64 {
        (0..n)
            .map(|j| (0..n).map(|i| mat[i][j].abs()).sum::<f64>())
            .fold(0.0_f64, f64::max)
    };
    let rcond = 1.0 / (norm1(a) * norm1(&inv));
    if !rcond.is_finite() || rcond < f64::EPSILON {
        return None;
    }
    let k = b[0].len();
    let mut out = vec![vec![0.0; k]; n];
    for i in 0..n {
        for j in 0..k {
            out[i][j] = (0..n).map(|l| inv[i][l] * b[l][j]).sum();
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// stats::arima objective functions
// ---------------------------------------------------------------------------

/// R `ARIMA_CSS` on the differenced series: `(ssq, nu)` from
/// observation `ncond` on, MA residuals before `ncond` taken as 0.
fn arima_css(w: &[f64], phi: &[f64], theta: &[f64], ncond: usize) -> (f64, usize) {
    let n = w.len();
    let mut resid = vec![0.0; n];
    let mut ssq = 0.0;
    let mut nu = 0usize;
    for l in ncond..n {
        let mut tmp = w[l];
        for (j, &ph) in phi.iter().enumerate() {
            tmp -= ph * w[l - j - 1];
        }
        let lim = (l - ncond).min(theta.len());
        for j in 0..lim {
            tmp -= theta[j] * resid[l - j - 1];
        }
        resid[l] = tmp;
        if !tmp.is_nan() {
            nu += 1;
            ssq += tmp * tmp;
        }
    }
    (ssq, nu)
}

struct Layout {
    p: usize,
    q: usize,
    cap_p: usize,
    cap_q: usize,
    m: usize,
    /// Value of the (differenced) mean/drift regressor, `None` when the
    /// candidate has no constant.
    creg: Option<f64>,
}

impl Layout {
    fn narma(&self) -> usize {
        self.p + self.q + self.cap_p + self.cap_q
    }
    fn split<'a>(&self, par: &'a [f64]) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
        let (p, q, cp, cq) = (self.p, self.q, self.cap_p, self.cap_q);
        (
            &par[0..p],
            &par[p..p + q],
            &par[p + q..p + q + cp],
            &par[p + q + cp..p + q + cp + cq],
        )
    }
    fn adjusted(&self, w: &[f64], par: &[f64]) -> Vec<f64> {
        match self.creg {
            Some(c) => {
                let beta = par[self.narma()];
                w.iter().map(|&v| v - beta * c).collect()
            }
            None => w.to_vec(),
        }
    }
    /// `C_ARIMA_transPars`: optional `partrans` on AR blocks, then the
    /// multiplicative seasonal expansion.
    fn trans_pars(&self, par: &[f64], trans: bool) -> (Vec<f64>, Vec<f64>) {
        let (ar, ma, sar, sma) = self.split(par);
        let (ar, sar) = if trans {
            (partrans(ar), partrans(sar))
        } else {
            (ar.to_vec(), sar.to_vec())
        };
        (
            expand_seasonal_ar(&ar, &sar, self.m),
            expand_seasonal_ma(ma, sma, self.m),
        )
    }
}

/// Fit one candidate exactly as `forecast:::myarima` -> `stats::arima`
/// would, on the already (d, D)-differenced series `w`. `creg` is the
/// differenced value of the constant regressor (1 for a mean or a d=1
/// drift, `m` for a D=1 drift) or `None` without a constant.
#[allow(clippy::too_many_arguments)]
pub(crate) fn r_arima_fit(
    x: &[f64],
    d: usize,
    cap_d: usize,
    p: usize,
    q: usize,
    cap_p: usize,
    cap_q: usize,
    m: usize,
    with_constant: bool,
    method: RMethod,
) -> Option<RArimaFit> {
    let m_full = m.max(1);
    let cap_d = if m_full > 1 { cap_d } else { 0 };
    // Differenced series (R: `diff(x, 1, d)` then `diff(., m, D)`).
    let mut w = x.to_vec();
    for _ in 0..d {
        w = (1..w.len()).map(|i| w[i] - w[i - 1]).collect();
    }
    for _ in 0..cap_d {
        if w.len() <= m_full {
            return None;
        }
        w = (m_full..w.len()).map(|i| w[i] - w[i - m_full]).collect();
    }
    let w = &w[..];
    let delta = delta_poly(d, cap_d, m_full);
    // myarima: a constant is a mean when d+D = 0 and a drift (time
    // index 1..n) when d+D = 1; never otherwise.
    let nd = d + cap_d;
    let with_constant = with_constant && nd <= 1;
    let creg = with_constant.then_some(if d == 0 && cap_d == 1 {
        m_full as f64
    } else {
        1.0
    });
    let xreg_at = |t: usize| -> f64 {
        if nd == 0 {
            1.0
        } else {
            (t + 1) as f64
        }
    };
    let m = if cap_p + cap_q > 0 { m_full } else { 1 };
    let lay = Layout {
        p,
        q,
        cap_p,
        cap_q,
        m,
        creg,
    };
    let n_used = w.len();
    let narma = lay.narma();
    let ncoef = narma + usize::from(creg.is_some());
    let ncond = p + cap_p * m;
    if n_used <= ncond + 1 {
        return None;
    }

    // init0 / parscale (lm on the differenced series for the regressor).
    let mut init = vec![0.0; ncoef];
    let mut parscale = vec![1.0; ncoef];
    if let Some(c) = creg {
        let (coef, se) = lm_single_constant(w, c)?;
        init[narma] = coef;
        parscale[narma] = 10.0 * se;
    }

    let css_obj = |par: &[f64]| -> f64 {
        let (phi, theta) = lay.trans_pars(par, false);
        let x = lay.adjusted(w, par);
        let (ssq, nu) = arima_css(&x, &phi, &theta, ncond);
        0.5 * (ssq / nu as f64).ln()
    };

    let nstar = n_used as f64;
    let npar = ncoef + 1;
    let k = npar as f64;
    let criteria = |aic: f64| -> Option<(f64, f64)> {
        if nstar - k - 1.0 <= 0.0 {
            return None;
        }
        let aicc = aic + 2.0 * k * (k + 1.0) / (nstar - k - 1.0);
        let bic = aic + k * (nstar.ln() - 2.0);
        (aicc.is_finite() && bic.is_finite()).then_some((aicc, bic))
    };

    if method == RMethod::Css {
        let (coef, hess) = if ncoef == 0 {
            (vec![], None)
        } else {
            let res = vmmin(&css_obj, &init, &parscale)?;
            let h = optimhess(&css_obj, &res.par, &parscale)?;
            (res.par, Some(h))
        };
        if let Some(h) = hess {
            // var <- solve(res$hessian * n.used); checkarima.
            let hn: Vec<Vec<f64>> = h
                .iter()
                .map(|r| r.iter().map(|v| v * n_used as f64).collect())
                .collect();
            let ident: Vec<Vec<f64>> = (0..ncoef)
                .map(|i| (0..ncoef).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                .collect();
            let var = r_solve(&hn, &ident)?;
            if (0..ncoef).any(|i| var[i][i].sqrt().is_nan()) {
                return None;
            }
        }
        let (phi, theta) = lay.trans_pars(&coef, false);
        let x = lay.adjusted(w, &coef);
        let (ssq, nu) = arima_css(&x, &phi, &theta, ncond);
        let sigma2 = ssq / nu as f64;
        if !(sigma2.is_finite() && sigma2 > 0.0) {
            return None;
        }
        let aic = nstar * sigma2.ln() + 2.0 * k;
        let (aicc, bic) = criteria(aic)?;
        let fit = finish(&lay, coef, f64::NAN, sigma2, aic, aicc, bic);
        return root_rule(fit, m);
    }

    // ---- CSS-ML ----
    if narma > 0 || creg.is_some() {
        let res = vmmin(&css_obj, &init, &parscale)?;
        if res.convergence == 0 {
            init = res.par;
        }
    }
    {
        let (ar, _, sar, _) = lay.split(&init);
        if !ar_check(ar) || !ar_check(sar) {
            return None; // "non-stationary (seasonal) AR part from CSS"
        }
    }
    // transform.pars: invpartrans on AR blocks, maInvert on MA blocks.
    {
        let (ar, ma, sar, sma) = lay.split(&init);
        let (ar, ma, sar, sma) = (
            invpartrans(ar),
            ma_invert(ma),
            invpartrans(sar),
            ma_invert(sma),
        );
        let mut next = Vec::with_capacity(ncoef);
        next.extend(ar);
        next.extend(ma);
        next.extend(sar);
        next.extend(sma);
        if creg.is_some() {
            next.push(init[narma]);
        }
        init = next;
    }
    if init.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let x_adjusted = |par: &[f64]| -> Vec<f64> {
        match creg {
            Some(_) => {
                let beta = par[narma];
                x.iter()
                    .enumerate()
                    .map(|(t, &v)| v - beta * xreg_at(t))
                    .collect()
            }
            None => x.to_vec(),
        }
    };
    let armafn = |par: &[f64]| -> f64 {
        let (phi, theta) = lay.trans_pars(par, true);
        let xa = x_adjusted(par);
        match arima_like(&xa, &phi, &theta, &delta) {
            Some((ssq, sumlog, nu)) => {
                let s2 = ssq / nu as f64;
                0.5 * (s2.ln() + sumlog / nu as f64)
            }
            None => f64::NAN,
        }
    };

    let (mut coef, mut value) = if ncoef == 0 {
        (vec![], armafn(&[]))
    } else {
        let res = vmmin(&armafn, &init, &parscale)?;
        (res.par, res.value)
    };

    if ncoef > 0 {
        // Final maInvert of the (untransformed) MA blocks; re-evaluate
        // when it changed anything (R's maxit = 0 re-run).
        let (p_, q_, cp_, cq_) = (p, q, cap_p, cap_q);
        let ma_new = ma_invert(&coef[p_..p_ + q_]);
        let sma_new = ma_invert(&coef[p_ + q_ + cp_..p_ + q_ + cp_ + cq_]);
        let mut changed = false;
        for (i, v) in ma_new.iter().enumerate() {
            if coef[p_ + i] != *v {
                changed = true;
                coef[p_ + i] = *v;
            }
        }
        for (i, v) in sma_new.iter().enumerate() {
            if coef[p_ + q_ + cp_ + i] != *v {
                changed = true;
                coef[p_ + q_ + cp_ + i] = *v;
            }
        }
        if changed {
            value = armafn(&coef);
        }
        let _ = cq_;
        // checkarima: var = A' (H * n.used)^-1 A, A = ARIMA_Gradtrans.
        let h = optimhess(&armafn, &coef, &parscale)?;
        let hn: Vec<Vec<f64>> = h
            .iter()
            .map(|r| r.iter().map(|v| v * n_used as f64).collect())
            .collect();
        let a = gradtrans(&lay, &coef, ncoef);
        let sol = r_solve(&hn, &a)?;
        for i in 0..ncoef {
            let vii: f64 = (0..ncoef).map(|l| a[l][i] * sol[l][i]).sum();
            if vii.sqrt().is_nan() {
                return None;
            }
        }
        // undoPars: AR blocks back to coefficient space.
        let (ar, _, sar, _) = lay.split(&coef);
        let (ar, sar) = (partrans(ar), partrans(sar));
        coef[..p].copy_from_slice(&ar);
        coef[p + q..p + q + cap_p].copy_from_slice(&sar);
    }
    if !value.is_finite() {
        return None;
    }
    let nf = n_used as f64;
    let loglik = -0.5 * (2.0 * nf * value + nf + nf * (2.0 * std::f64::consts::PI).ln());
    let aic = -2.0 * loglik + 2.0 * k;
    let (aicc, bic) = criteria(aic)?;
    let sigma2 = {
        let (phi, theta) = lay.trans_pars(&coef, false);
        arima_like(&x_adjusted(&coef), &phi, &theta, &delta)
            .map_or(f64::NAN, |(ssq, _, _)| ssq / nf)
    };
    let fit = finish(&lay, coef, loglik, sigma2, aic, aicc, bic);
    root_rule(fit, m)
}

// ---------------------------------------------------------------------------
// R's state-space likelihood: getQ0 (Gardner et al. 1980, AS154) and
// ARIMA_Like (diffuse-prior Kalman filter on the undifferenced series)
// ---------------------------------------------------------------------------

/// AS154 `inclu2` (Givens-style update of `d`, `rbar`, `thetab`).
fn inclu2(
    xnext: &[f64],
    xrow: &mut [f64],
    mut ynext: f64,
    d: &mut [f64],
    rbar: &mut [f64],
    thetab: &mut [f64],
) {
    let np = xnext.len();
    xrow.copy_from_slice(xnext);
    let mut ithisr = 0usize;
    for i in 0..np {
        if xrow[i] != 0.0 {
            let xi = xrow[i];
            let di = d[i];
            let dpi = di + xi * xi;
            d[i] = dpi;
            let cbar = di / dpi;
            let sbar = xi / dpi;
            for k in (i + 1)..np {
                let xk = xrow[k];
                let rbthis = rbar[ithisr];
                xrow[k] = xk - xi * rbthis;
                rbar[ithisr] = cbar * rbthis + sbar * xk;
                ithisr += 1;
            }
            let xk = ynext;
            ynext = xk - xi * thetab[i];
            thetab[i] = cbar * thetab[i] + sbar * xk;
            if di == 0.0 {
                return;
            }
        } else {
            ithisr += np - i - 1;
        }
    }
}

/// R's C `getQ0` (`SSinit = "Gardner1980"`): the `r x r` initial
/// state covariance, row-major `r*r`.
fn get_q0(phi: &[f64], theta: &[f64]) -> Vec<f64> {
    let p = phi.len();
    let q = theta.len();
    let r = p.max(q + 1);
    let np = r * (r + 1) / 2;
    let nrbar = np * (np - 1) / 2;
    let mut v = vec![0.0; np];
    let mut ind = 0usize;
    for j in 0..r {
        let vj = if j == 0 {
            1.0
        } else if j - 1 < q {
            theta[j - 1]
        } else {
            0.0
        };
        for i in j..r {
            let vi = if i == 0 {
                1.0
            } else if i - 1 < q {
                theta[i - 1]
            } else {
                0.0
            };
            v[ind] = vi * vj;
            ind += 1;
        }
    }
    let mut pm = vec![0.0; (r * r).max(np)];
    if r == 1 {
        pm[0] = if p == 0 {
            1.0
        } else {
            1.0 / (1.0 - phi[0] * phi[0])
        };
        pm.truncate(1);
        return pm;
    }
    if p > 0 {
        let mut rbar = vec![0.0; nrbar];
        let mut thetab = vec![0.0; np];
        let mut xnext = vec![0.0; np];
        let mut xrow = vec![0.0; np];
        let mut ind = 0usize;
        let mut ind1: i64 = -1;
        let npr = np - r;
        let npr1 = npr + 1;
        let mut indj = npr;
        let mut ind2: i64 = npr as i64 - 1;
        for j in 0..r {
            let phij = if j < p { phi[j] } else { 0.0 };
            xnext[indj] = 0.0;
            indj += 1;
            let mut indi = npr1 + j;
            for i in j..r {
                let ynext = v[ind];
                ind += 1;
                let phii = if i < p { phi[i] } else { 0.0 };
                if j != r - 1 {
                    xnext[indj] = -phii;
                    if i != r - 1 {
                        xnext[indi] -= phij;
                        ind1 += 1;
                        xnext[ind1 as usize] = -1.0;
                    }
                }
                xnext[npr] = -phii * phij;
                ind2 += 1;
                if ind2 as usize >= np {
                    ind2 = 0;
                }
                xnext[ind2 as usize] += 1.0;
                inclu2(
                    &xnext,
                    &mut xrow,
                    ynext,
                    &mut pm[..np],
                    &mut rbar,
                    &mut thetab,
                );
                xnext[ind2 as usize] = 0.0;
                if i != r - 1 {
                    xnext[indi] = 0.0;
                    indi += 1;
                    xnext[ind1 as usize] = 0.0;
                }
            }
        }
        let mut ithisr: i64 = nrbar as i64 - 1;
        let mut im: i64 = np as i64 - 1;
        for i in 0..np {
            let mut bi = thetab[im as usize];
            let mut jm: i64 = np as i64 - 1;
            for _ in 0..i {
                bi -= rbar[ithisr as usize] * pm[jm as usize];
                ithisr -= 1;
                jm -= 1;
            }
            pm[im as usize] = bi;
            im -= 1;
        }
        // re-order p
        xnext[..r].copy_from_slice(&pm[npr..npr + r]);
        let mut ind: i64 = np as i64 - 1;
        let mut ind1: i64 = npr as i64 - 1;
        for _ in 0..npr {
            pm[ind as usize] = pm[ind1 as usize];
            ind -= 1;
            ind1 -= 1;
        }
        pm[..r].copy_from_slice(&xnext[..r]);
    } else {
        let mut indn = np;
        let mut ind = np;
        for i in 0..r {
            for j in 0..=i {
                ind -= 1;
                pm[ind] = v[ind];
                if j != 0 {
                    indn -= 1;
                    pm[ind] += pm[indn];
                }
            }
        }
    }
    // unpack to a full (column-major in R; symmetric so layout-agnostic)
    let mut ind = np;
    for i in (1..r).rev() {
        for j in (i..r).rev() {
            ind -= 1;
            pm[r * i + j] = pm[ind];
        }
    }
    for i in 0..r.saturating_sub(1) {
        for j in (i + 1)..r {
            pm[i + r * j] = pm[j + r * i];
        }
    }
    pm.truncate(r * r);
    pm
}

/// R's C `ARIMA_Like` with `sUP = 0`: returns `(ssq, sumlog, nu)`.
/// `x` is the undifferenced (regression-adjusted) series; `delta` the
/// differencing polynomial (R's `Delta`), `kappa = 1e6` on its states.
fn arima_like(
    x: &[f64],
    phi: &[f64],
    theta_in: &[f64],
    delta: &[f64],
) -> Option<(f64, f64, usize)> {
    arima_filter(x, phi, theta_in, delta).map(|f| (f.ssq, f.sumlog, f.nu))
}

/// Result of one `ARIMA_Like` pass, including the filtered state
/// covariance `P_{n|n}` that R leaves behind in `mod$P` (the state
/// `KalmanForecast` starts from).
struct FilterState {
    ssq: f64,
    sumlog: f64,
    nu: usize,
    /// `sum(residuals^2)` over *all* observations, where R's
    /// `residuals = resid / sqrt(gain)` (also the diffuse ones with
    /// `gain >= 1e4` that `ssq` skips).
    rss_all: f64,
    /// `rd x rd` filtered state covariance, column-major (`i + rd*j`),
    /// in units of `sigma2`.
    pmat: Vec<f64>,
}

/// Body of R's `ARIMA_Like` (see [`arima_like`]), also returning the
/// filtered state covariance.
fn arima_filter(x: &[f64], phi: &[f64], theta_in: &[f64], delta: &[f64]) -> Option<FilterState> {
    let p = phi.len();
    let q0 = theta_in.len();
    let r = p.max(q0 + 1);
    let d = delta.len();
    let rd = r + d;
    // makeARIMA pads theta to r - 1.
    let mut theta = theta_in.to_vec();
    if theta.len() < r - 1 {
        theta.resize(r - 1, 0.0);
    }
    let q = theta.len();
    let mut a = vec![0.0; rd];
    let mut pmat = vec![0.0; rd * rd];
    let mut pnew = vec![0.0; rd * rd];
    // Pn initialisation (upARIMA / makeARIMA), column-major index i + rd*j.
    if r > 1 {
        let q0m = get_q0(phi, theta_in);
        for i in 0..r {
            for j in 0..r {
                pnew[i + rd * j] = q0m[i + r * j];
            }
        }
    } else {
        pnew[0] = if p > 0 {
            1.0 / (1.0 - phi[0] * phi[0])
        } else {
            1.0
        };
    }
    for i in 0..d {
        pnew[(r + i) + rd * (r + i)] = 1e6;
    }
    let mut anew = vec![0.0; rd];
    let mut mvec = vec![0.0; rd];
    let mut mm = vec![0.0; rd * rd];
    let (mut ssq, mut sumlog, mut nu) = (0.0_f64, 0.0_f64, 0usize);
    let mut rss_all = 0.0_f64;
    for (l, &yl) in x.iter().enumerate() {
        for i in 0..r {
            let mut tmp = if i < r - 1 { a[i + 1] } else { 0.0 };
            if i < p {
                tmp += phi[i] * a[0];
            }
            anew[i] = tmp;
        }
        if d > 0 {
            for i in (r + 1)..rd {
                anew[i] = a[i - 1];
            }
            let mut tmp = a[0];
            for i in 0..d {
                tmp += delta[i] * a[r + i];
            }
            anew[r] = tmp;
        }
        if l > 0 {
            if d == 0 {
                for i in 0..r {
                    let vi = if i == 0 {
                        1.0
                    } else if i - 1 < q {
                        theta[i - 1]
                    } else {
                        0.0
                    };
                    for j in 0..r {
                        let mut tmp = if j == 0 {
                            vi
                        } else if j - 1 < q {
                            vi * theta[j - 1]
                        } else {
                            0.0
                        };
                        if i < p && j < p {
                            tmp += phi[i] * phi[j] * pmat[0];
                        }
                        if i < r - 1 && j < r - 1 {
                            tmp += pmat[i + 1 + r * (j + 1)];
                        }
                        if i < p && j < r - 1 {
                            tmp += phi[i] * pmat[j + 1];
                        }
                        if j < p && i < r - 1 {
                            tmp += phi[j] * pmat[i + 1];
                        }
                        pnew[i + r * j] = tmp;
                    }
                }
            } else {
                for i in 0..r {
                    for j in 0..rd {
                        let mut tmp = 0.0;
                        if i < p {
                            tmp += phi[i] * pmat[rd * j];
                        }
                        if i < r - 1 {
                            tmp += pmat[i + 1 + rd * j];
                        }
                        mm[i + rd * j] = tmp;
                    }
                }
                for j in 0..rd {
                    let mut tmp = pmat[rd * j];
                    for k in 0..d {
                        tmp += delta[k] * pmat[r + k + rd * j];
                    }
                    mm[r + rd * j] = tmp;
                }
                for i in 1..d {
                    for j in 0..rd {
                        mm[r + i + rd * j] = pmat[r + i - 1 + rd * j];
                    }
                }
                for i in 0..r {
                    for j in 0..rd {
                        let mut tmp = 0.0;
                        if i < p {
                            tmp += phi[i] * mm[j];
                        }
                        if i < r - 1 {
                            tmp += mm[rd * (i + 1) + j];
                        }
                        pnew[j + rd * i] = tmp;
                    }
                }
                for j in 0..rd {
                    let mut tmp = mm[j];
                    for k in 0..d {
                        tmp += delta[k] * mm[rd * (r + k) + j];
                    }
                    pnew[rd * r + j] = tmp;
                }
                for i in 1..d {
                    for j in 0..rd {
                        pnew[rd * (r + i) + j] = mm[rd * (r + i - 1) + j];
                    }
                }
                for i in 0..=q {
                    let vi = if i == 0 { 1.0 } else { theta[i - 1] };
                    for j in 0..=q {
                        pnew[i + rd * j] += vi * if j == 0 { 1.0 } else { theta[j - 1] };
                    }
                }
            }
        }
        let mut resid = yl - anew[0];
        for i in 0..d {
            resid -= delta[i] * anew[r + i];
        }
        for i in 0..rd {
            let mut tmp = pnew[i];
            for j in 0..d {
                tmp += pnew[i + (r + j) * rd] * delta[j];
            }
            mvec[i] = tmp;
        }
        let mut gain = mvec[0];
        for j in 0..d {
            gain += delta[j] * mvec[r + j];
        }
        if gain < 1e4 {
            nu += 1;
            ssq += resid * resid / gain;
            sumlog += gain.ln();
        }
        rss_all += resid * resid / gain;
        for i in 0..rd {
            a[i] = anew[i] + mvec[i] * resid / gain;
        }
        for i in 0..rd {
            for j in 0..rd {
                pmat[i + j * rd] = pnew[i + j * rd] - mvec[i] * mvec[j] / gain;
            }
        }
    }
    (ssq.is_finite() && sumlog.is_finite() && nu > 0).then_some(FilterState {
        ssq,
        sumlog,
        nu,
        rss_all,
        pmat,
    })
}

/// R's C `KalmanFore` (`stats::KalmanForecast`) restricted to the
/// variance recursion: starting from the filtered covariance `pmat`
/// of the `makeARIMA` state-space form (`T` with the AR block and the
/// differencing polynomial `delta` folded in, `V = R R'` with
/// `R = (1, theta, 0)`, `Z = (1, 0, ..., delta)`, `h = 0`), returns
/// `Z P_{n+l|n} Z'` for `l = 1..=horizon`, in units of `sigma2`.
fn kalman_forecast_var(
    phi: &[f64],
    theta_in: &[f64],
    delta: &[f64],
    mut pmat: Vec<f64>,
    horizon: usize,
) -> Vec<f64> {
    let p = phi.len();
    let r = p.max(theta_in.len() + 1);
    let d = delta.len();
    let rd = r + d;
    debug_assert_eq!(pmat.len(), rd * rd);
    // T (column-major `i + rd*k`).
    let mut t = vec![0.0; rd * rd];
    for (i, &v) in phi.iter().enumerate() {
        t[i] = v;
    }
    for i in 0..r.saturating_sub(1) {
        t[i + rd * (i + 1)] = 1.0;
    }
    if d > 0 {
        t[r] = 1.0;
        for (j, &v) in delta.iter().enumerate() {
            t[r + rd * (r + j)] = v;
        }
        for i in 1..d {
            t[(r + i) + rd * (r + i - 1)] = 1.0;
        }
    }
    // V = R R'.
    let mut rvec = vec![0.0; rd];
    rvec[0] = 1.0;
    for (i, &v) in theta_in.iter().enumerate() {
        rvec[i + 1] = v;
    }
    let mut z = vec![0.0; rd];
    z[0] = 1.0;
    for (j, &v) in delta.iter().enumerate() {
        z[r + j] = v;
    }
    let mut mm = vec![0.0; rd * rd];
    let mut out = Vec::with_capacity(horizon);
    for _ in 0..horizon {
        for i in 0..rd {
            for j in 0..rd {
                let mut tmp = 0.0;
                for k in 0..rd {
                    tmp += t[i + rd * k] * pmat[k + rd * j];
                }
                mm[i + rd * j] = tmp;
            }
        }
        for i in 0..rd {
            for j in 0..rd {
                let mut tmp = rvec[i] * rvec[j];
                for k in 0..rd {
                    tmp += mm[i + rd * k] * t[j + rd * k];
                }
                pmat[i + rd * j] = tmp;
            }
        }
        let mut var = 0.0;
        for i in 0..rd {
            for j in 0..rd {
                var += z[i] * z[j] * pmat[i + j * rd];
            }
        }
        out.push(var);
    }
    out
}

/// R's ARIMA prediction variances (`forecast::forecast.Arima` ->
/// `predict.Arima`): one `ARIMA_Like` pass over the regression-adjusted,
/// undifferenced series `x` gives the standardized residuals and the
/// filtered state; `KalmanForecast` then propagates the state.
///
/// `sigma2` is the `forecast` package's (`Arima` / `auto.arima`'s
/// `myarima` overwrite `stats::arima`'s `ssq / n.used`):
/// `sum(residuals^2) / (nstar - npar + 1)` with `nstar = n.used =
/// n - length(Delta)` and `npar = ncoef + 1`, i.e. `rss / (n.used -
/// ncoef)` where `ncoef` counts the estimated ARMA + regression
/// coefficients. Returns
/// `(sigma2, [sigma2 * Z P_{n+l|n} Z'; l = 1..=horizon])`, i.e. the
/// squared standard errors `forecast::forecast.Arima` puts into its
/// intervals. `phi`/`theta` are the expanded (multiplicative-seasonal)
/// AR/MA polynomials; `delta` comes from [`delta_poly`].
pub(crate) fn r_prediction_variances(
    x: &[f64],
    phi: &[f64],
    theta: &[f64],
    delta: &[f64],
    ncoef: usize,
    horizon: usize,
) -> Option<(f64, Vec<f64>)> {
    let dof = x
        .len()
        .checked_sub(delta.len() + ncoef)
        .filter(|&n| n > 0)?;
    let state = arima_filter(x, phi, theta, delta)?;
    let sigma2 = state.rss_all / dof as f64;
    if !sigma2.is_finite() || sigma2 < 0.0 {
        return None;
    }
    let var = kalman_forecast_var(phi, theta, delta, state.pmat, horizon);
    let var: Vec<f64> = var.into_iter().map(|v| v * sigma2).collect();
    var.iter().all(|v| v.is_finite()).then_some((sigma2, var))
}

/// R's `Delta` for `(1-B)^d (1-B^m)^D`, as the coefficient vector of
/// `-Delta[-1]` (so `x_t = sum delta_i x_{t-i} + w_t`).
pub(crate) fn delta_poly(d: usize, cap_d: usize, m: usize) -> Vec<f64> {
    let conv = |a: &[f64], b: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                out[i + j] += ai * bj;
            }
        }
        out
    };
    let mut del = vec![1.0];
    for _ in 0..d {
        del = conv(&del, &[1.0, -1.0]);
    }
    let mut seas = vec![0.0; m + 1];
    seas[0] = 1.0;
    seas[m] = -1.0;
    for _ in 0..cap_d {
        del = conv(&del, &seas);
    }
    del[1..].iter().map(|v| -v).collect()
}

/// `lm(dx ~ dxreg - 1)` for a single constant regressor column
/// (`dxreg = c`), reproducing R's LINPACK path (`dqrdc2` Householder
/// QR + `dqrsl`) operation by operation, then `summary.lm`'s standard
/// error (`rss` accumulated in extended precision like R's `sum()`;
/// Neumaier compensation here). Returns `(coef, se)`.
///
/// The arithmetic order matters: `se` feeds `optim`'s `parscale`, and
/// on ill-conditioned likelihood surfaces a last-ulp difference in it
/// is enough to send BFGS down a different trajectory than R's
/// (measured on a near-deterministic AR series, 11-12).
fn lm_single_constant(y: &[f64], c: f64) -> Option<(f64, f64)> {
    let n = y.len();
    if n < 2 {
        return None;
    }
    // dqrdc2: nrmxl = dnrm2(x), sign of x(1), x <- x/nrmxl, x(1) += 1.
    let nrmxl = (n as f64 * c * c).sqrt().copysign(c);
    let mut x = vec![c * (1.0 / nrmxl); n];
    x[0] += 1.0;
    let qraux = x[0];
    let r11 = -nrmxl;
    // dqrsl qty: t = -ddot(x, y)/x(1); qty = y + t*x.
    let dot: f64 = x.iter().zip(y).fold(0.0, |acc, (a, b)| acc + a * b);
    let t = -dot / qraux;
    let qty: Vec<f64> = y.iter().zip(&x).map(|(yi, xi)| yi + t * xi).collect();
    let coef = qty[0] / r11;
    // Residuals: Q applied to (0, qty[1..]).
    let mut rsd = qty;
    rsd[0] = 0.0;
    let dot: f64 = x.iter().zip(&rsd).fold(0.0, |acc, (a, b)| acc + a * b);
    let t = -dot / qraux;
    for (ri, xi) in rsd.iter_mut().zip(&x) {
        *ri += t * xi;
    }
    // Neumaier-compensated sum of squares (R's long-double `sum`).
    let (mut sum, mut comp) = (0.0_f64, 0.0_f64);
    for r in &rsd {
        let v = r * r;
        let tsum = sum + v;
        if sum.abs() >= v.abs() {
            comp += (sum - tsum) + v;
        } else {
            comp += (v - tsum) + sum;
        }
        sum = tsum;
    }
    let rss = sum + comp;
    let resvar = rss / (n as f64 - 1.0);
    let se = ((1.0 / (r11 * r11)) * resvar).sqrt();
    (coef.is_finite() && se.is_finite() && se > 0.0).then_some((coef, se))
}

/// `C_ARIMA_Gradtrans` (forward differences, eps = 1e-3) as a dense
/// `A[i][j] = d new_j / d raw_i` matrix.
fn gradtrans(lay: &Layout, raw: &[f64], ncoef: usize) -> Vec<Vec<f64>> {
    const EPS: f64 = 1e-3;
    let mut a: Vec<Vec<f64>> = (0..ncoef)
        .map(|i| (0..ncoef).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let mut block = |off: usize, len: usize| {
        if len == 0 {
            return;
        }
        let mut w1: Vec<f64> = raw[off..off + len].to_vec();
        let w2 = partrans(&w1);
        for i in 0..len {
            w1[i] += EPS;
            let w3 = partrans(&w1);
            for j in 0..len {
                a[off + i][off + j] = (w3[j] - w2[j]) / EPS;
            }
            w1[i] -= EPS;
        }
    };
    block(0, lay.p);
    block(lay.p + lay.q, lay.cap_p);
    a
}

fn finish(
    lay: &Layout,
    coef: Vec<f64>,
    loglik: f64,
    sigma2: f64,
    aic: f64,
    aicc: f64,
    bic: f64,
) -> RArimaFit {
    let (ar, ma, sar, sma) = if coef.len() >= lay.narma() {
        let (a, b, c, d) = lay.split(&coef);
        (a.to_vec(), b.to_vec(), c.to_vec(), d.to_vec())
    } else {
        (vec![], vec![], vec![], vec![])
    };
    let constant = if lay.creg.is_some() {
        coef[lay.narma()]
    } else {
        0.0
    };
    RArimaFit {
        loglik,
        sigma2,
        aic,
        aicc,
        bic,
        ar,
        ma,
        sar,
        sma,
        constant,
    }
}

/// `myarima`'s `minroot < 1.01` rule on the AR and MA polynomials
/// (checked per factor, see `seasonal_roots_outside`).
fn root_rule(fit: RArimaFit, m: usize) -> Option<RArimaFit> {
    if !seasonal_roots_outside(&fit.ar, &fit.sar, m, 1.01) {
        return None;
    }
    let ma_neg: Vec<f64> = fit.ma.iter().map(|v| -v).collect();
    let sma_neg: Vec<f64> = fit.sma.iter().map(|v| -v).collect();
    seasonal_roots_outside(&ma_neg, &sma_neg, m, 1.01).then_some(fit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partrans_roundtrip() {
        let phi = [0.5, -0.3, 0.1];
        let raw = invpartrans(&phi);
        let back = partrans(&raw);
        for (a, b) in phi.iter().zip(&back) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn ma_invert_reflects_roots() {
        // 1 + 2z has root -0.5 (inside) -> 1 + 0.5z.
        let out = ma_invert(&[2.0]);
        assert!((out[0] - 0.5).abs() < 1e-12);
        // Already invertible: unchanged.
        assert_eq!(ma_invert(&[0.3, 0.1]), vec![0.3, 0.1]);
        // 1 - 2.5z + z^2 = (1-2z)(1-0.5z): root 0.5 inside -> (1-0.5z)^2.
        let out = ma_invert(&[-2.5, 1.0]);
        assert!((out[0] + 1.0).abs() < 1e-9, "{out:?}");
        assert!((out[1] - 0.25).abs() < 1e-9, "{out:?}");
    }

    #[test]
    fn ar_check_basic() {
        assert!(ar_check(&[0.5]));
        assert!(!ar_check(&[1.0]));
        assert!(ar_check(&[0.9465, -0.1449]));
        assert!(!ar_check(&[1.2, -0.1]));
    }

    /// ARIMA(0,1,0): after the diffuse level state is resolved, the
    /// h-step variance is exactly `h` (in sigma2 units).
    #[test]
    fn kalman_forecast_random_walk_variance_is_h() {
        let x: Vec<f64> = (0..50)
            .map(|t| (t as f64 * 0.7).sin() * 3.0 + t as f64)
            .collect();
        let delta = delta_poly(1, 0, 1);
        let state = arima_filter(&x, &[], &[], &delta).unwrap();
        let var = kalman_forecast_var(&[], &[], &delta, state.pmat, 5);
        for (h, v) in var.iter().enumerate() {
            assert!((v - (h + 1) as f64).abs() < 1e-6, "h={}: {v}", h + 1);
        }
    }

    /// AR(1) without differencing: `var_h = sum_{j<h} phi^(2j)`.
    #[test]
    fn kalman_forecast_ar1_matches_psi_weights() {
        let phi = 0.6_f64;
        let x: Vec<f64> = (0..80).map(|t| ((t * 7919) % 13) as f64 - 6.0).collect();
        let state = arima_filter(&x, &[phi], &[], &[]).unwrap();
        let var = kalman_forecast_var(&[phi], &[], &[], state.pmat, 6);
        for (h, v) in var.iter().enumerate() {
            let expect: f64 = (0..=h).map(|j| phi.powi(2 * j as i32)).sum();
            assert!((v - expect).abs() < 1e-12, "h={}: {v} vs {expect}", h + 1);
        }
    }
}
