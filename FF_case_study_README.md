# Phase 2 Case Study: Inference on Asset Returns with a Single Dataset
**Date:** 2025-08-12

## Dataset (single source)
Use the **Ken French Data Library** monthly data:
- **Factors:** Fama–French 5 factors + Momentum (monthly) including the risk-free rate.
- **Portfolios:** 25 Portfolios Formed on Size × Book-to-Market (monthly returns).

> Put the CSVs in a local folder, e.g. `data/`. Typical files are:
> - `F-F_Research_Data_5_Factors_2x3.csv` (or ZIP with CSV inside)
> - `F-F_Momentum_Factor.csv` (sometimes named differently)
> - `25_Portfolios_5x5.CSV`

All tasks below operate on this dataset only. You will compute **excess returns**, perform **classical inference** (point estimates, CIs, tests), use **bootstrap**/**block bootstrap**, and fit **likelihood models**. The questions are connected; try to preserve code re-use (helper functions, clean API).

---

## Deliverables
- A single notebook (or script) that:
  - Loads and reconciles the data into a tidy monthly DataFrame with columns: `date`, `portfolio`, `ret`, `rf`, `mktrf`, `smb`, `hml`, `rmw`, `cma`, `umd`.
  - Implements inference utilities (means, CIs, tests, bootstrap, MLE).
  - Produces clean tables and plots to support conclusions.
- A short **write-up** (markdown cells) answering each question with both math and empirical findings.

---

## Challenge Set (connected questions)

### A) Data sanity + excess returns
1. Clean & align the factor and portfolio series, monthly frequency. Show summary stats.
2. Compute **excess returns** for each portfolio: `ret_excess = ret - rf`.
3. Visualize distributions (histogram + QQ plot) for a few portfolios. Comment on symmetry, tails, and normality.

**Math**: define sample mean/variance, unbiasedness; recall CLT assumptions.

---

### B) Estimation & Confidence Intervals (single portfolio)
4. For one chosen portfolio (e.g., `Small-High`), estimate the **mean excess return** and **95% CI** under:
   - (i) Normal theory (t-based CI)
   - (ii) **IID bootstrap**
   - (iii) **Moving block bootstrap** (block length b in [6, 12]).
5. Compare widths and coverage via **simulation** using your fitted Normal model.

**Math**: derive t-based CI; define bootstrap CI (percentile/BCa optional). Discuss serial dependence impact and why block bootstrap helps.

---

### C) Multiple Testing across 25 portfolios
6. Test H0: mu_i <= 0 vs H1: mu_i > 0 for each of the 25 portfolios (one-sided t-test on mean excess return).
7. Control for multiplicity using:
   - **FWER**: Bonferroni and Holm.
   - **FDR**: Benjamini–Hochberg.
   Report discoveries and adjusted p-values. Interpret in plain language.

**Math**: state definitions of FWER and FDR, and the proofs/ideas behind Bonferroni monotonicity and BH’s step-up procedure.

---

### D) Sharpe ratio inference
8. Estimate annualized **Sharpe ratios** for each portfolio. Give **95% CIs** using:
   - (i) **Delta method** approximation,
   - (ii) **Bootstrap** (IID and block). Compare results.
9. For your chosen portfolio, test if Sharpe > 0.25 (annualized) at 5% level.

**Math**: derive delta-method variance of S_hat = mu_hat / sigma_hat. Discuss finite-sample bias and heavy tails.

---

### E) Structural breaks (crisis period)
10. Pre-register a single break date (e.g., 2008-09). For each portfolio, test change in mean excess return before vs after (two-sample t-test; Welch if variances differ).
11. Repeat for variance change (Levene/Brown–Forsythe or bootstrap).
12. Summarize the **proportion** of portfolios with significant breaks; apply BH-FDR control.

**Math**: derive two-sample t-statistic; discuss effect sizes vs p-values.

---

### F) Likelihood modeling & LR tests (heavy tails)
13. For a selected portfolio, fit **Normal** and **Student-t** by **MLE**. Report mu_hat, sigma_hat^2, nu_hat and standard errors (observed information).
14. Perform a **Likelihood Ratio test** of Normal vs Student-t (t nests Normal as nu -> infinity); use a **parametric bootstrap** to approximate the LR null because of boundary issues.
15. Revisit CI for mean under the t model; compare with earlier CIs.

**Math**: write the log-likelihoods; compute gradients; discuss regularity conditions and boundary cases.

---

### G) Power & sample size for positive alpha
16. Suppose you target a mean excess return of **3% annualized**. For a given volatility (estimated from data), compute the **sample size in months** needed to achieve 80% power at 5% significance (one-sided t-test). Do it (i) **analytically** using the noncentral t approximation and (ii) via **Monte Carlo** simulation.
17. Plot power curves vs sample size and effect size.

**Math**: derive power formula for one-sample t; show the noncentrality parameter and its relationship to effect size.

---

### H) (Optional) Bayesian aside
18. Use a Normal–Inverse-Gamma prior for (mu, sigma^2). Compute the posterior and the posterior probability **P(mu > 0)** for your chosen portfolio.
19. Compare posterior intervals to frequentist CIs.

**Math**: conjugacy derivation; link to shrinkage.

---

## Quality bar (“what good looks like”)
- Functions are **vectorized**, tested, documented.
- Clear separation of **math derivations** (markdown) and **code**.
- All tests include **assumptions** and **diagnostics**.
- Reproducible random seeds; tables and plots are labeled.
- Conclusions are crisp and quantitative (effect sizes, not just p-values).
