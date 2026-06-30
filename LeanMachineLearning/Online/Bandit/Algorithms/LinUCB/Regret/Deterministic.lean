/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: OpenAI, Fawad Haider
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.LinUCB.Regret.CoordinateConfidence

/-!
# LinUCB Regret: Deterministic Elliptical-Potential Bounds

This module is part of the LinUCB regret proof stack.
-/

@[expose] public section

open MeasureTheory ProbabilityTheory Filter Real Finset Learning

open scoped ENNReal NNReal Matrix MatrixOrder

namespace Bandits

variable {K d : ℕ}

namespace LinUCB

variable {hK : 0 < K} {reg : ℝ} {β : ℕ → ℝ} {x : Fin K → Feature d}
  {ν : Kernel (Fin K) ℝ} [IsMarkovKernel ν]
  {Ω : Type*} {mΩ : MeasurableSpace Ω}
  {P : Measure Ω} [IsProbabilityMeasure P]
  {A : ℕ → Ω → Fin K} {R : ℕ → Ω → ℝ}
  {n : ℕ} {ω : Ω}

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- At horizon zero, the log-determinant regret bound is deterministic. -/
lemma regret_le_textbook_logDet_bound_of_zero_horizon
    (β : ℕ → ℝ) (hn : n = 0) :
    regret ν A n ω ≤
      (if n = 0 then 0 else gap ν (A 0 ω)) +
        2 * (√((n : ℝ) * β n) * √(ellipticalPotential A reg x n ω)) := by
  subst n
  simp [regret_eq_sum_gap]

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- In zero feature dimension under linear realizability, the log-determinant regret bound is
deterministic. -/
lemma regret_le_textbook_logDet_bound_of_linear_dim_eq_zero [Nonempty (Fin K)]
    (β : ℕ → ℝ) {θ : Feature d} (hd : d = 0) (h_linear : LinearMeanModel ν x θ) :
    regret ν A n ω ≤
      (if n = 0 then 0 else gap ν (A 0 ω)) +
        2 * (√((n : ℝ) * β n) * √(ellipticalPotential A reg x n ω)) := by
  rw [regret_eq_zero_of_linear_dim_eq_zero (A := A) (ν := ν) (x := x) (θ := θ)
    (n := n) (ω := ω) hd h_linear]
  have hinit_nonneg : 0 ≤ if n = 0 then 0 else gap ν (A 0 ω) := by
    by_cases hn : n = 0
    · simp [hn]
    · simp [hn, gap_nonneg]
  have hbonus_nonneg :
      0 ≤ 2 * (√((n : ℝ) * β n) * √(ellipticalPotential A reg x n ω)) := by
    positivity
  linarith

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Under linear realizability, a nonpositive feature-norm bound makes the log-determinant regret
bound deterministic. -/
lemma regret_le_textbook_logDet_bound_of_featureSqNormBound_nonpos [Nonempty (Fin K)]
    (β : ℕ → ℝ) {L2 : ℝ} {θ : Feature d}
    (hL2 : FeatureSqNormBound x L2) (hL2_nonpos : L2 ≤ 0)
    (h_linear : LinearMeanModel ν x θ) :
    regret ν A n ω ≤
      (if n = 0 then 0 else gap ν (A 0 ω)) +
        2 * (√((n : ℝ) * β n) * √(ellipticalPotential A reg x n ω)) := by
  rw [regret_eq_zero_of_linear_featureSqNormBound_nonpos (A := A) (ν := ν)
    (x := x) (n := n) (ω := ω) hL2 hL2_nonpos h_linear]
  have hinit_nonneg : 0 ≤ if n = 0 then 0 else gap ν (A 0 ω) := by
    by_cases hn : n = 0
    · simp [hn]
    · simp [hn, gap_nonneg]
  have hbonus_nonneg :
      0 ≤ 2 * (√((n : ℝ) * β n) * √(ellipticalPotential A reg x n ω)) := by
    positivity
  linarith

/-- Almost surely, cumulative regret is bounded by the initial gap plus
`2 * √(n * β n) * √W` whenever the squared LinUCB widths are almost surely bounded by `W` and `β`
is nonnegative and monotone. -/
lemma regret_ae_le_initial_add_sqrt_nat_mul_beta_width_bound [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_best : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) n ω)
    (h_arm : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      estimatedReward A R reg x (A n ω) n ω -
        √(β (n + 1)) * width A reg x (A n ω) n ω ≤ (ν (A n ω))[id])
    (hβ : ∀ t, 0 ≤ β (t + 1)) (hβ_mono : Monotone β) (W : ℝ)
    (hW : ∀ᵐ ω ∂P, widthSqSum A reg x n ω ≤ W) :
    ∀ᵐ ω ∂P,
      regret ν A n ω ≤
        (∑ t ∈ range n, if t = 0 then gap ν (A 0 ω) else 0) +
          2 * (√((n : ℝ) * β n) * √W) := by
  exact regret_ae_le_initial_add_sqrt_bounds (A := A) (R := R) (reg := reg) (β := β)
    (x := x) (ν := ν) (n := n) h h_best h_arm hβ ((n : ℝ) * β n) W
    (beta_sum_le_nat_mul_of_monotone (β := β) (n := n) hβ_mono hβ) hW

/-- Almost surely, cumulative regret is bounded by the simplified initial-gap term plus
`2 * √(n * β n) * √W` whenever the squared LinUCB widths are almost surely bounded by `W` and `β`
is nonnegative and monotone. -/
lemma regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_width_bound [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_best : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) n ω)
    (h_arm : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      estimatedReward A R reg x (A n ω) n ω -
        √(β (n + 1)) * width A reg x (A n ω) n ω ≤ (ν (A n ω))[id])
    (hβ : ∀ t, 0 ≤ β (t + 1)) (hβ_mono : Monotone β) (W : ℝ)
    (hW : ∀ᵐ ω ∂P, widthSqSum A reg x n ω ≤ W) :
    ∀ᵐ ω ∂P,
      regret ν A n ω ≤
        (if n = 0 then 0 else gap ν (A 0 ω)) + 2 * (√((n : ℝ) * β n) * √W) := by
  filter_upwards [regret_ae_le_initial_add_sqrt_nat_mul_beta_width_bound (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_best h_arm hβ hβ_mono W hW
    ] with ω h_regret
  simpa [initial_gap_sum_eq (A := A) (ν := ν) (n := n) (ω := ω)] using h_regret

/-- Almost surely, cumulative regret is bounded by the simplified initial-gap term plus
`2 * √(n * β n) * √W` whenever a history-level quadratic-form bound supplies the
elliptical-potential input. -/
lemma regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_history_quadratic_bound [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_best : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) n ω)
    (h_arm : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      estimatedReward A R reg x (A n ω) n ω -
        √(β (n + 1)) * width A reg x (A n ω) n ω ≤ (ν (A n ω))[id])
    (hβ : ∀ t, 0 ≤ β (t + 1)) (hβ_mono : Monotone β) (W : ℝ)
    (h_quad_nonneg : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω))
    (hW : ∀ᵐ ω ∂P, historyQuadraticWidthSum A R reg x n ω ≤ W) :
    ∀ᵐ ω ∂P,
      regret ν A n ω ≤
        (if n = 0 then 0 else gap ν (A 0 ω)) + 2 * (√((n : ℝ) * β n) * √W) := by
  exact regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_width_bound (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_best h_arm hβ hβ_mono W
    (widthSqSum_ae_le_of_history_quadratic_width_sum_ae_le (A := A) (R := R)
      (reg := reg) (x := x) (n := n) (P := P) (W := W) h_quad_nonneg hW)

/-- Almost surely, cumulative regret is bounded by the simplified initial-gap term plus
`2 * √(n * β n) * √W` whenever the packaged history-level quadratic-width input holds almost
surely.

This is the theorem a history-level elliptical-potential lemma can feed into directly. -/
lemma regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_history_quadratic_width_bound
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_best : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) n ω)
    (h_arm : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      estimatedReward A R reg x (A n ω) n ω -
        √(β (n + 1)) * width A reg x (A n ω) n ω ≤ (ν (A n ω))[id])
    (hβ : ∀ t, 0 ≤ β (t + 1)) (hβ_mono : Monotone β) (W : ℝ)
    (h_bound : ∀ᵐ ω ∂P, HistoryQuadraticWidthBound A R reg x n ω W) :
    ∀ᵐ ω ∂P,
      regret ν A n ω ≤
        (if n = 0 then 0 else gap ν (A 0 ω)) + 2 * (√((n : ℝ) * β n) * √W) := by
  exact regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_width_bound (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_best h_arm hβ hβ_mono W
    (widthSqSum_ae_le_of_history_quadratic_width_bound_ae (A := A) (R := R)
      (reg := reg) (x := x) (n := n) (P := P) (W := W) h_bound)

/-- Almost surely, cumulative regret is bounded by the simplified initial-gap term plus
`2 * √(n * β n) * √W` whenever a capped history-level quadratic-width sum bound holds almost
surely and every positive-time quadratic width form is almost surely nonnegative and at most `1`.

This is the direct interface for the common capped form of the elliptical-potential lemma. -/
lemma regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_capped_history_quadratic_bound
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_best : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) n ω)
    (h_arm : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      estimatedReward A R reg x (A n ω) n ω -
        √(β (n + 1)) * width A reg x (A n ω) n ω ≤ (ν (A n ω))[id])
    (hβ : ∀ t, 0 ≤ β (t + 1)) (hβ_mono : Monotone β) (W : ℝ)
    (h_quad_nonneg : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω))
    (h_quad_le_one : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω) ≤ 1)
    (hW : ∀ᵐ ω ∂P, historyCappedQuadraticWidthSum A R reg x n ω ≤ W) :
    ∀ᵐ ω ∂P,
      regret ν A n ω ≤
        (if n = 0 then 0 else gap ν (A 0 ω)) + 2 * (√((n : ℝ) * β n) * √W) := by
  exact regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_width_bound (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_best h_arm hβ hβ_mono W
    (widthSqSum_ae_le_of_capped_history_quadratic_width_sum_ae_le (A := A) (R := R)
      (reg := reg) (x := x) (n := n) (P := P) (W := W) h_quad_nonneg h_quad_le_one hW)

/-- Almost surely, cumulative regret is bounded by the simplified initial-gap term plus
`2 * √(n * β n) * √W` whenever a capped process-level quadratic-width sum bound holds almost
surely and every positive-time process-level quadratic width form is almost surely nonnegative and
at most `1`.

This is the direct interface for an elliptical-potential lemma stated using the process-level design
matrices `designMatrix A reg x t ω`. -/
lemma regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_capped_quadratic_bound
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_best : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) n ω)
    (h_arm : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      estimatedReward A R reg x (A n ω) n ω -
        √(β (n + 1)) * width A reg x (A n ω) n ω ≤ (ν (A n ω))[id])
    (hβ : ∀ t, 0 ≤ β (t + 1)) (hβ_mono : Monotone β) (W : ℝ)
    (h_quad_nonneg : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm A reg x (A t ω) t ω)
    (h_quad_le_one : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      widthQuadraticForm A reg x (A t ω) t ω ≤ 1)
    (hW : ∀ᵐ ω ∂P, cappedQuadraticWidthSum A reg x n ω ≤ W) :
    ∀ᵐ ω ∂P,
      regret ν A n ω ≤
        (if n = 0 then 0 else gap ν (A 0 ω)) + 2 * (√((n : ℝ) * β n) * √W) := by
  exact regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_width_bound (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_best h_arm hβ hβ_mono W
    (widthSqSum_ae_le_of_capped_quadratic_width_sum_ae_le (A := A) (reg := reg)
      (x := x) (n := n) (P := P) (W := W) h_quad_nonneg h_quad_le_one hW)

/-- Almost surely, cumulative regret is bounded by the simplified initial-gap term plus
`2 * √(n * β n) * √W` whenever the packaged process-level capped quadratic-width input holds
almost surely.

This is the compact theorem a process-level elliptical-potential lemma should feed into directly. -/
lemma regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_capped_quadratic_width_bound
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_best : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) n ω)
    (h_arm : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      estimatedReward A R reg x (A n ω) n ω -
        √(β (n + 1)) * width A reg x (A n ω) n ω ≤ (ν (A n ω))[id])
    (hβ : ∀ t, 0 ≤ β (t + 1)) (hβ_mono : Monotone β) (W : ℝ)
    (h_bound : ∀ᵐ ω ∂P, CappedQuadraticWidthBound A reg x n ω W) :
    ∀ᵐ ω ∂P,
      regret ν A n ω ≤
        (if n = 0 then 0 else gap ν (A 0 ω)) + 2 * (√((n : ℝ) * β n) * √W) := by
  exact regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_width_bound (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_best h_arm hβ hβ_mono W
    (widthSqSum_ae_le_of_capped_quadratic_width_bound_ae (A := A) (reg := reg)
      (x := x) (n := n) (P := P) (W := W) h_bound)

/-- Almost surely, cumulative regret is bounded by the simplified initial-gap term plus
`2 * √(n * β n) * √W` whenever the textbook capped quadratic-width sum is almost surely bounded
by `W`.

This version follows the proof structure of *Bandit Algorithms*, Theorem 19.2: optimism gives the
width bound, bounded instantaneous gaps give the cap, monotonicity of `β` moves all confidence
radii to `β n`, and Cauchy-Schwarz turns the sum into the square root of the capped quadratic-width
sum. -/
lemma regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_capped_sum_bound
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_best : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) n ω)
    (h_arm : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      estimatedReward A R reg x (A n ω) n ω -
        √(β (n + 1)) * width A reg x (A n ω) n ω ≤ (ν (A n ω))[id])
    (h_gap_two : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 → gap ν (A t ω) ≤ 2)
    (hβ_schedule : BetaSchedule β) (W : ℝ)
    (h_quad_nonneg : ∀ᵐ ω ∂P, ∀ t, t ∈ range n →
      0 ≤ widthQuadraticForm A reg x (A t ω) t ω)
    (hW : ∀ᵐ ω ∂P, cappedQuadraticWidthSum A reg x n ω ≤ W) :
    ∀ᵐ ω ∂P,
      regret ν A n ω ≤
        (if n = 0 then 0 else gap ν (A 0 ω)) + 2 * (√((n : ℝ) * β n) * √W) := by
  by_cases hn : n = 0
  · subst n
    exact Filter.Eventually.of_forall fun ω ↦ by simp [regret]
  have hβn_nonneg : 0 ≤ β n :=
    hβ_schedule.nonneg_of_ne_zero hn
  filter_upwards [forall_gap_arm_le_two_mul_width h h_best h_arm, h_gap_two, h_quad_nonneg, hW]
    with ω h_gap_widthω h_gap_twoω h_quad_nonnegω hWω
  have h_quad_pos : ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm A reg x (A t ω) t ω := by
    intro t ht _ht0
    exact h_quad_nonnegω t ht
  have h_gap_capped : ∀ t, t ∈ range n → t ≠ 0 →
      gap ν (A t ω) ≤
        2 * (√(β n) * √(min 1 (widthQuadraticForm A reg x (A t ω) t ω))) := by
    intro t ht ht0
    have hβ_le : β (t + 1) ≤ β n :=
      hβ_schedule.monotone (Nat.succ_le_iff.mpr (mem_range.mp ht))
    have ht_pos : 0 < t := Nat.pos_of_ne_zero ht0
    have hn_pos : 0 < n := Nat.lt_trans ht_pos (mem_range.mp ht)
    have hn_one : 1 ≤ n := Nat.succ_le_iff.mpr hn_pos
    have hβn_one : 1 ≤ β n := hβ_schedule.one.trans (hβ_schedule.monotone hn_one)
    exact gap_le_two_mul_sqrt_beta_mul_sqrt_min_widthQuadraticForm (A := A)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (ω := ω) (t := t)
      (h_gap_twoω t ht ht0) (h_gap_widthω t ht0) hβ_le hβn_one
  have h_regret :
      regret ν A n ω ≤
        (∑ t ∈ range n, if t = 0 then gap ν (A 0 ω) else 0) +
          2 * (√((n : ℝ) * β n) * √(cappedQuadraticWidthSum A reg x n ω)) :=
    regret_le_initial_add_sqrt_nat_mul_beta_capped_sum (A := A) (reg := reg)
      (β := β) (x := x) (ν := ν) (n := n) (ω := ω) hβn_nonneg h_quad_pos
      h_gap_capped
  simpa [initial_gap_sum_eq (A := A) (ν := ν) (n := n) (ω := ω)] using
    regret_le_initial_add_sqrt_nat_mul_beta_of_capped_sum_le (A := A) (reg := reg)
      (β := β) (x := x) (ν := ν) (n := n) (ω := ω) W h_regret hWω

/-- Almost surely, on the LinUCB confidence event, cumulative regret is bounded by the simplified
initial-gap term plus `2 * √(n * β n) * √W` whenever the textbook capped quadratic-width sum is
almost surely bounded by `W`.

This is the good-event form of the deterministic regret argument. It separates the algorithmic
regret proof from the future concentration theorem: a later self-normalized concentration result
should prove that `LinUCBConfidenceEvent` holds with high probability, and this theorem converts
that event into the regret bound. -/
lemma regret_ae_imp_le_initial_gap_add_sqrt_nat_mul_beta_capped_sum_bound
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_gap_two : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 → gap ν (A t ω) ≤ 2)
    (hβ_schedule : BetaSchedule β) (W : ℝ)
    (h_quad_nonneg : ∀ᵐ ω ∂P, ∀ t, t ∈ range n →
      0 ≤ widthQuadraticForm A reg x (A t ω) t ω)
    (hW : ∀ᵐ ω ∂P, cappedQuadraticWidthSum A reg x n ω ≤ W) :
    ∀ᵐ ω ∂P,
      LinUCBConfidenceEvent A R reg β x ν ω →
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) + 2 * (√((n : ℝ) * β n) * √W) := by
  by_cases hn : n = 0
  · subst n
    exact Filter.Eventually.of_forall fun ω _h_confω ↦ by simp [regret]
  have hβn_nonneg : 0 ≤ β n :=
    hβ_schedule.nonneg_of_ne_zero hn
  filter_upwards [forall_index_le_index_arm h (bestArm ν), h_gap_two, h_quad_nonneg, hW] with
    ω h_indexω h_gap_twoω h_quad_nonnegω hWω h_confω
  have h_quad_pos : ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm A reg x (A t ω) t ω := by
    intro t ht _ht0
    exact h_quad_nonnegω t ht
  have h_gap_capped : ∀ t, t ∈ range n → t ≠ 0 →
      gap ν (A t ω) ≤
        2 * (√(β n) * √(min 1 (widthQuadraticForm A reg x (A t ω) t ω))) := by
    intro t ht ht0
    have h_gap_width :
        gap ν (A t ω) ≤ 2 * (√(β (t + 1)) * width A reg x (A t ω) t ω) :=
      gap_arm_le_two_mul_width (A := A) (R := R) (reg := reg) (β := β)
        (x := x) (ν := ν) (n := t) (ω := ω) (h_confω.best t ht0)
        (h_confω.arm t ht0) (h_indexω t ht0)
    have hβ_le : β (t + 1) ≤ β n :=
      hβ_schedule.monotone (Nat.succ_le_iff.mpr (mem_range.mp ht))
    have ht_pos : 0 < t := Nat.pos_of_ne_zero ht0
    have hn_pos : 0 < n := Nat.lt_trans ht_pos (mem_range.mp ht)
    have hn_one : 1 ≤ n := Nat.succ_le_iff.mpr hn_pos
    have hβn_one : 1 ≤ β n := hβ_schedule.one.trans (hβ_schedule.monotone hn_one)
    exact gap_le_two_mul_sqrt_beta_mul_sqrt_min_widthQuadraticForm (A := A)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (ω := ω) (t := t)
      (h_gap_twoω t ht ht0) h_gap_width hβ_le hβn_one
  have h_regret :
      regret ν A n ω ≤
        (∑ t ∈ range n, if t = 0 then gap ν (A 0 ω) else 0) +
          2 * (√((n : ℝ) * β n) * √(cappedQuadraticWidthSum A reg x n ω)) :=
    regret_le_initial_add_sqrt_nat_mul_beta_capped_sum (A := A) (reg := reg)
      (β := β) (x := x) (ν := ν) (n := n) (ω := ω) hβn_nonneg h_quad_pos
      h_gap_capped
  simpa [initial_gap_sum_eq (A := A) (ν := ν) (n := n) (ω := ω)] using
    regret_le_initial_add_sqrt_nat_mul_beta_of_capped_sum_le (A := A) (reg := reg)
      (β := β) (x := x) (ν := ν) (n := n) (ω := ω) W h_regret hWω

/-- Horizon-local good-event form of the deterministic LinUCB regret argument.

Compared with `regret_ae_imp_le_initial_gap_add_sqrt_nat_mul_beta_capped_sum_bound`, this theorem
only asks for confidence inequalities at positive times inside `range n`, which is exactly what an
`n`-round regret proof uses. -/
lemma regret_ae_imp_le_initial_gap_add_sqrt_nat_mul_beta_capped_sum_bound_upTo
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_gap_two : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 → gap ν (A t ω) ≤ 2)
    (hβ_schedule : BetaSchedule β) (W : ℝ)
    (h_quad_nonneg : ∀ᵐ ω ∂P, ∀ t, t ∈ range n →
      0 ≤ widthQuadraticForm A reg x (A t ω) t ω)
    (hW : ∀ᵐ ω ∂P, cappedQuadraticWidthSum A reg x n ω ≤ W) :
    ∀ᵐ ω ∂P,
      LinUCBConfidenceEventUpTo A R reg β x ν n ω →
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) + 2 * (√((n : ℝ) * β n) * √W) := by
  by_cases hn : n = 0
  · subst n
    exact Filter.Eventually.of_forall fun ω _h_confω ↦ by simp [regret]
  have hβn_nonneg : 0 ≤ β n :=
    hβ_schedule.nonneg_of_ne_zero hn
  filter_upwards [forall_index_le_index_arm h (bestArm ν), h_gap_two, h_quad_nonneg, hW] with
    ω h_indexω h_gap_twoω h_quad_nonnegω hWω h_confω
  have h_quad_pos : ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm A reg x (A t ω) t ω := by
    intro t ht _ht0
    exact h_quad_nonnegω t ht
  have h_gap_capped : ∀ t, t ∈ range n → t ≠ 0 →
      gap ν (A t ω) ≤
        2 * (√(β n) * √(min 1 (widthQuadraticForm A reg x (A t ω) t ω))) := by
    intro t ht ht0
    have h_gap_width :
        gap ν (A t ω) ≤ 2 * (√(β (t + 1)) * width A reg x (A t ω) t ω) :=
      gap_arm_le_two_mul_width (A := A) (R := R) (reg := reg) (β := β)
        (x := x) (ν := ν) (n := t) (ω := ω)
        (h_confω.best t ht ht0) (h_confω.arm t ht ht0) (h_indexω t ht0)
    have hβ_le : β (t + 1) ≤ β n :=
      hβ_schedule.monotone (Nat.succ_le_iff.mpr (mem_range.mp ht))
    have ht_pos : 0 < t := Nat.pos_of_ne_zero ht0
    have hn_pos : 0 < n := Nat.lt_trans ht_pos (mem_range.mp ht)
    have hn_one : 1 ≤ n := Nat.succ_le_iff.mpr hn_pos
    have hβn_one : 1 ≤ β n := hβ_schedule.one.trans (hβ_schedule.monotone hn_one)
    exact gap_le_two_mul_sqrt_beta_mul_sqrt_min_widthQuadraticForm (A := A)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (ω := ω) (t := t)
      (h_gap_twoω t ht ht0) h_gap_width hβ_le hβn_one
  have h_regret :
      regret ν A n ω ≤
        (∑ t ∈ range n, if t = 0 then gap ν (A 0 ω) else 0) +
          2 * (√((n : ℝ) * β n) * √(cappedQuadraticWidthSum A reg x n ω)) :=
    regret_le_initial_add_sqrt_nat_mul_beta_capped_sum (A := A) (reg := reg)
      (β := β) (x := x) (ν := ν) (n := n) (ω := ω) hβn_nonneg h_quad_pos
      h_gap_capped
  simpa [initial_gap_sum_eq (A := A) (ν := ν) (n := n) (ω := ω)] using
    regret_le_initial_add_sqrt_nat_mul_beta_of_capped_sum_le (A := A) (reg := reg)
      (β := β) (x := x) (ν := ν) (n := n) (ω := ω) W h_regret hWω

/-- Horizon-local good-event regret bound with the realized capped quadratic-width sum.

This is the same deterministic argument as
`regret_ae_imp_le_initial_gap_add_sqrt_nat_mul_beta_capped_sum_bound_upTo`, but it keeps the
sample-path capped width sum on the right-hand side instead of immediately replacing it by a
deterministic bound `W`. This is the natural input to the log-determinant
elliptical-potential lemma. -/
lemma regret_ae_imp_le_initial_gap_add_sqrt_nat_mul_beta_capped_sum_upTo
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_gap_two : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 → gap ν (A t ω) ≤ 2)
    (hβ_schedule : BetaSchedule β)
    (h_quad_nonneg : ∀ᵐ ω ∂P, ∀ t, t ∈ range n →
      0 ≤ widthQuadraticForm A reg x (A t ω) t ω) :
    ∀ᵐ ω ∂P,
      LinUCBConfidenceEventUpTo A R reg β x ν n ω →
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) *
              √(cappedQuadraticWidthSum A reg x n ω)) := by
  by_cases hn : n = 0
  · subst n
    exact Filter.Eventually.of_forall fun ω _h_confω ↦ by simp [regret]
  have hβn_nonneg : 0 ≤ β n :=
    hβ_schedule.nonneg_of_ne_zero hn
  filter_upwards [forall_index_le_index_arm h (bestArm ν), h_gap_two, h_quad_nonneg] with
    ω h_indexω h_gap_twoω h_quad_nonnegω h_confω
  have h_quad_pos : ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm A reg x (A t ω) t ω := by
    intro t ht _ht0
    exact h_quad_nonnegω t ht
  have h_gap_capped : ∀ t, t ∈ range n → t ≠ 0 →
      gap ν (A t ω) ≤
        2 * (√(β n) * √(min 1 (widthQuadraticForm A reg x (A t ω) t ω))) := by
    intro t ht ht0
    have h_gap_width :
        gap ν (A t ω) ≤ 2 * (√(β (t + 1)) * width A reg x (A t ω) t ω) :=
      gap_arm_le_two_mul_width (A := A) (R := R) (reg := reg) (β := β)
        (x := x) (ν := ν) (n := t) (ω := ω)
        (h_confω.best t ht ht0) (h_confω.arm t ht ht0) (h_indexω t ht0)
    have hβ_le : β (t + 1) ≤ β n :=
      hβ_schedule.monotone (Nat.succ_le_iff.mpr (mem_range.mp ht))
    have ht_pos : 0 < t := Nat.pos_of_ne_zero ht0
    have hn_pos : 0 < n := Nat.lt_trans ht_pos (mem_range.mp ht)
    have hn_one : 1 ≤ n := Nat.succ_le_iff.mpr hn_pos
    have hβn_one : 1 ≤ β n := hβ_schedule.one.trans (hβ_schedule.monotone hn_one)
    exact gap_le_two_mul_sqrt_beta_mul_sqrt_min_widthQuadraticForm (A := A)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (ω := ω) (t := t)
      (h_gap_twoω t ht ht0) h_gap_width hβ_le hβn_one
  have h_regret :
      regret ν A n ω ≤
        (∑ t ∈ range n, if t = 0 then gap ν (A 0 ω) else 0) +
          2 * (√((n : ℝ) * β n) * √(cappedQuadraticWidthSum A reg x n ω)) :=
    regret_le_initial_add_sqrt_nat_mul_beta_capped_sum (A := A) (reg := reg)
      (β := β) (x := x) (ν := ν) (n := n) (ω := ω) hβn_nonneg h_quad_pos
      h_gap_capped
  simpa [initial_gap_sum_eq (A := A) (ν := ν) (n := n) (ω := ω)] using h_regret

/-- Almost surely, cumulative regret is bounded by the simplified initial-gap term plus the
feature-budget elliptical-potential term
`2 * √(n * β n) * √(2 * d * log(1 + n L² / (reg d)))`.

The determinant/trace input is isolated as `h_ratio_of_trace`: it proves that the trace budget
implies the displayed determinant-ratio bound. -/
lemma regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_featureSqNorm_budget_bound
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_best : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) n ω)
    (h_arm : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      estimatedReward A R reg x (A n ω) n ω -
        √(β (n + 1)) * width A reg x (A n ω) n ω ≤ (ν (A n ω))[id])
    (hβ : ∀ t, 0 ≤ β (t + 1)) (hβ_mono : Monotone β)
    (hreg_pos : 0 < reg) (hd : d ≠ 0)
    (L2 : ℝ)
    (hL2 : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → featureSqNorm x (A t ω) ≤ L2)
    (hL2_le_reg : L2 ≤ reg)
    (h_ratio_of_trace : ∀ ω,
      designTrace A reg x n ω ≤ reg * (d : ℝ) + (n : ℝ) * L2 →
        designDetRatio A reg x n ω ≤
          ((reg * (d : ℝ) + (n : ℝ) * L2) / (reg * (d : ℝ))) ^ d) :
    ∀ᵐ ω ∂P,
      regret ν A n ω ≤
        (if n = 0 then 0 else gap ν (A 0 ω)) +
          2 * (√((n : ℝ) * β n) *
            √(2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ))))) := by
  exact regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_capped_quadratic_width_bound
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_best
    h_arm hβ hβ_mono
    (2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ))))
    (cappedQuadraticWidthBound_ae_of_reg_ne_zero_det_update_featureSqNorm_budget_bound'
      (A := A) (reg := reg) (x := x) (n := n) (P := P) hreg_pos.ne' hd
      (widthQuadraticForm_ae_nonneg_of_reg_nonneg (A := A) (reg := reg) (x := x)
        (n := n) (P := P) hreg_pos.le)
      (widthQuadraticForm_ae_le_one_of_featureSqNorm_ae_le (A := A) (reg := reg)
        (x := x) (n := n) (P := P)
        (WidthQuadraticFormLeFeatureSqNormDivReg.of_reg_pos (A := A) (reg := reg)
          (x := x) hreg_pos)
        hreg_pos hL2 hL2_le_reg)
      L2 hL2 h_ratio_of_trace)

/-- Almost surely, cumulative regret is bounded by the feature-budget elliptical-potential term
when the determinant/trace input is stated as the determinant upper bound
`det(V_n) ≤ ((reg * d + n * L²) / d) ^ d`. -/
lemma regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_featureSqNorm_budget_bound_of_designDet_le
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_best : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) n ω)
    (h_arm : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      estimatedReward A R reg x (A n ω) n ω -
        √(β (n + 1)) * width A reg x (A n ω) n ω ≤ (ν (A n ω))[id])
    (hβ : ∀ t, 0 ≤ β (t + 1)) (hβ_mono : Monotone β)
    (hreg_pos : 0 < reg) (hd : d ≠ 0)
    (L2 : ℝ)
    (hL2 : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → featureSqNorm x (A t ω) ≤ L2)
    (hL2_le_reg : L2 ≤ reg)
    (hdet_of_trace : ∀ ω,
      designTrace A reg x n ω ≤ reg * (d : ℝ) + (n : ℝ) * L2 →
        designDet A reg x n ω ≤
          ((reg * (d : ℝ) + (n : ℝ) * L2) / (d : ℝ)) ^ d) :
    ∀ᵐ ω ∂P,
      regret ν A n ω ≤
        (if n = 0 then 0 else gap ν (A 0 ω)) +
          2 * (√((n : ℝ) * β n) *
            √(2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ))))) := by
  exact regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_capped_quadratic_width_bound
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_best
    h_arm hβ hβ_mono
    (2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ))))
    (cappedQuadraticWidthBound_ae_of_reg_pos_det_update_featureSqNorm_budget_bound_of_designDet_le
      (A := A) (reg := reg) (x := x) (n := n) (P := P) hreg_pos hd
      L2 hL2 hL2_le_reg hdet_of_trace)

/-- Almost surely, cumulative regret is bounded by the feature-budget elliptical-potential term
when the determinant/trace input is the reusable PSD matrix determinant/trace comparison
`det(M) ≤ (trace(M) / d) ^ d`. -/
lemma regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_of_matrix_det_trace_bound
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_best : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) n ω)
    (h_arm : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      estimatedReward A R reg x (A n ω) n ω -
        √(β (n + 1)) * width A reg x (A n ω) n ω ≤ (ν (A n ω))[id])
    (hβ : ∀ t, 0 ≤ β (t + 1)) (hβ_mono : Monotone β)
    (hreg_pos : 0 < reg) (hd : d ≠ 0)
    (L2 : ℝ)
    (hL2 : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → featureSqNorm x (A t ω) ≤ L2)
    (hL2_le_reg : L2 ≤ reg)
    (hdet_trace : MatrixDetLeTraceAveragePow d) :
    ∀ᵐ ω ∂P,
      regret ν A n ω ≤
        (if n = 0 then 0 else gap ν (A 0 ω)) +
          2 * (√((n : ℝ) * β n) *
            √(2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ))))) := by
  exact regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_capped_quadratic_width_bound
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_best
    h_arm hβ hβ_mono
    (2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ))))
    (cappedQuadraticWidthBound_ae_of_matrix_det_trace_bound
      (A := A) (reg := reg) (x := x) (n := n) (P := P) hreg_pos hd
      L2 hL2 hL2_le_reg hdet_trace)

end LinUCB

end Bandits
