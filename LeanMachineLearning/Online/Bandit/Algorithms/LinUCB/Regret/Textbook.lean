/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: OpenAI, Fawad Haider
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.LinUCB.Regret.Deterministic

/-!
# LinUCB Regret: Textbook Good-Event Bounds

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

/-- Textbook-shaped finite-action LinUCB regret theorem on the horizon-local confidence event.

This is the finite-horizon good-event form closest to the proof in *Bandit Algorithms*: to control
regret up to horizon `n`, the theorem only assumes optimism and lower-confidence validity for
positive times `t ∈ range n`. The deterministic algorithm/max-index argument and the
elliptical-potential bound are proved internally. The remaining probabilistic task is to prove the
horizon-local confidence event with high probability. -/
lemma regret_ae_imp_le_textbook_finite_action_upTo
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) :
    ∀ᵐ ω ∂P,
      LinUCBConfidenceEventUpTo A R reg β x ν n ω →
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) *
              √(2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ))))) := by
  by_cases hd : d = 0
  · subst d
    exact Filter.Eventually.of_forall fun ω h_confω ↦ by
      simpa using regret_le_initial_gap_of_confidenceUpTo_dim_eq_zero
        (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
        (ω := ω) (d := 0) rfl h_confω
  · have h_gap_two : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 → gap ν (A t ω) ≤ 2 := by
      filter_upwards [gap_ae_le_of_GapBound (A := A) (ν := ν) (n := n) (P := P)
        2 (gapBound_two_of_meanRewardBound_neg_one_one (ν := ν) h_mean_bound)] with
        ω h_gapω
      intro t ht _ht0
      exact h_gapω t ht
    exact regret_ae_imp_le_initial_gap_add_sqrt_nat_mul_beta_capped_sum_bound_upTo
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
      h_gap_two hβ_schedule
      (2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ))))
      (widthQuadraticForm_ae_nonneg_of_reg_nonneg (A := A) (reg := reg) (x := x)
        (n := n) (P := P) hreg_pos.le)
      (cappedQuadraticWidthSum_ae_le_featureSqNorm_budget_of_matrix_det_trace_bound
        (A := A) (reg := reg) (x := x) (n := n) (P := P) hreg_pos hd L2
        (featureSqNorm_ae_le_of_featureSqNormBound (A := A) (x := x) (n := n)
          (P := P) L2 hL2)
        matrixDetLeTraceAveragePow)

/-- Textbook-shaped finite-action LinUCB regret theorem on the confidence event.

This theorem is the good-event form closest to the finite-action LinUCB proof in
*Bandit Algorithms*: after the deterministic algorithm/max-index argument and the elliptical
potential bound are proved, the only remaining probabilistic input is whether the confidence event
holds on a sample path.

* `h_mean_bound` bounds every arm's mean reward in `[-1, 1]`;
* `hβ_schedule` states that the confidence-radius schedule starts at least at one and is monotone;
* `hL2` is the uniform finite-action feature bound `‖x_a‖₂² ≤ L2`.

The conclusion is an almost-sure implication: on almost every sample path, if
`LinUCBConfidenceEvent` holds, then the displayed regret bound holds. A future self-normalized
concentration theorem should prove that this confidence event has high probability for a concrete
textbook choice of `β`.

The displayed bound is the standard Cauchy-Schwarz plus elliptical-potential expression
`2 * sqrt(n * β_n) * sqrt(2 d log(1 + n L² / (reg d)))`, with one extra initial gap because this
formalization lets the deterministic algorithm play its default initial arm at time zero. -/
lemma regret_ae_imp_le_textbook_finite_action
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) :
    ∀ᵐ ω ∂P,
      LinUCBConfidenceEvent A R reg β x ν ω →
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) *
              √(2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ))))) := by
  filter_upwards [regret_ae_imp_le_textbook_finite_action_upTo (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_mean_bound hβ_schedule
    hreg_pos L2 hL2] with ω h_regret h_confω
  exact h_regret (LinUCBConfidenceEvent.toUpTo (A := A) (R := R) (reg := reg)
    (β := β) (x := x) (ν := ν) (n := n) (ω := ω) h_confω)

/-- The deterministic textbook LinUCB bonus term
`2 * sqrt(n * β_n) * sqrt(2 d log(1 + n L² / (reg d)))`.

The final finite-action theorem keeps this as a named expression so probability statements can use
a deterministic right-hand side instead of repeating the full formula. -/
noncomputable def textbookRegretBonus (reg : ℝ) (β : ℕ → ℝ) (L2 : ℝ) (n : ℕ) : ℝ :=
  2 * (√((n : ℝ) * β n) *
    √(2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ)))))

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Closed-form deterministic finite-horizon textbook LinUCB regret bonus.

This specializes `textbookRegretBonus` to the fixed finite-horizon textbook radius
`textbookLinUCBConfidenceRadius`. The generic `textbookRegretBonus` is still useful for abstract
beta schedules; this name is the theorem-facing expression closest to the textbook statement. -/
noncomputable def textbookLinUCBRegretBonus
    (d : ℕ) (reg S2 : ℝ) (σ2 : ℝ≥0) (L2 : ℝ) (n : ℕ) (δ : ℝ) : ℝ :=
  2 * (√((n : ℝ) * textbookLinUCBConfidenceRadius d reg S2 σ2 L2 n δ) *
    √(2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ)))))

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Applying the generic textbook regret bonus to the fixed-horizon textbook beta schedule reduces
to the closed-form textbook LinUCB regret bonus. -/
lemma textbookRegretBonus_textbookLinUCBBeta_eq
    (d : ℕ) (reg S2 : ℝ) (σ2 : ℝ≥0) (L2 : ℝ) (n : ℕ) (δ : ℝ) :
    textbookRegretBonus (d := d) reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) L2 n =
      textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ := by
  simp [textbookRegretBonus, textbookLinUCBRegretBonus,
    textbookLinUCBBeta_at_horizon d reg S2 σ2 L2 n δ]

omit [IsMarkovKernel ν] in
lemma regret_le_textbookRegretDet_bound_of_linear_dim_eq_zero [Nonempty (Fin K)]
    (β : ℕ → ℝ) (L2 : ℝ) {θ : Feature d} (hd : d = 0)
    (h_linear : LinearMeanModel ν x θ) :
    regret ν A n ω ≤
      (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n := by
  rw [regret_eq_zero_of_linear_dim_eq_zero (A := A) (ν := ν) (x := x) (θ := θ)
    (n := n) (ω := ω) hd h_linear]
  have hinit_nonneg : 0 ≤ if n = 0 then 0 else (2 : ℝ) := by
    by_cases hn : n = 0 <;> simp [hn]
  have hbonus_nonneg : 0 ≤ textbookRegretBonus (d := d) reg β L2 n := by
    unfold textbookRegretBonus
    exact mul_nonneg (by norm_num) (mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _))
  linarith

lemma regret_ae_imp_le_textbook_finite_action_deterministic_bound_upTo
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) :
    ∀ᵐ ω ∂P,
      LinUCBConfidenceEventUpTo A R reg β x ν n ω →
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n := by
  filter_upwards [regret_ae_imp_le_textbook_finite_action_upTo (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_mean_bound hβ_schedule
    hreg_pos L2 hL2] with ω h_regret h_confω
  refine (h_regret h_confω).trans ?_
  simpa [textbookRegretBonus] using
    add_le_add_right
      (initialGapTerm_le_two_of_meanRewardBound_neg_one_one (A := A) (ν := ν)
        (n := n) (ω := ω) h_mean_bound)
      (2 * (√((n : ℝ) * β n) *
        √(2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ))))))

lemma probReal_confidenceEventUpTo_le_textbook_regret_bound_deterministic
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) :
    P.real {ω | LinUCBConfidenceEventUpTo A R reg β x ν n ω} ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  filter_upwards [regret_ae_imp_le_textbook_finite_action_deterministic_bound_upTo
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2] with ω h_regret h_confω
  exact h_regret h_confω

/-- High-probability wrapper for the deterministic textbook finite-action LinUCB regret bound,
consuming only a finite-horizon confidence event. -/
lemma probReal_textbook_regret_bound_deterministic_ge_of_confidenceEventUpTo_ge
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) {δ : ℝ}
    (h_conf_prob :
      1 - δ ≤ P.real {ω | LinUCBConfidenceEventUpTo A R reg β x ν n ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  exact h_conf_prob.trans
    (probReal_confidenceEventUpTo_le_textbook_regret_bound_deterministic (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_mean_bound hβ_schedule
      hreg_pos L2 hL2)

/-- Short exported high-probability endpoint for the finite-action LinUCB regret proof.

This is the UCB-style theorem interface: once a concentration theorem proves the horizon-local
LinUCB confidence event with probability at least `1 - δ`, the textbook-shaped deterministic regret
bound also holds with probability at least `1 - δ`. -/
-- ANCHOR: LinUCB.regret_bound_ge
lemma regret_bound_ge_of_confidenceEventUpTo_ge
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) {δ : ℝ}
    (h_conf_prob :
      1 - δ ≤ P.real {ω | LinUCBConfidenceEventUpTo A R reg β x ν n ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  exact probReal_textbook_regret_bound_deterministic_ge_of_confidenceEventUpTo_ge
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2 h_conf_prob
-- ANCHOR_END: LinUCB.regret_bound_ge

/-- High-probability textbook regret bound from the self-normalized prediction-confidence event. -/
lemma probReal_textbook_regret_bound_deterministic_ge_of_selfNormalizedConfidenceEventUpTo_ge
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) {δ : ℝ}
    (h_self_prob :
      1 - δ ≤ P.real {ω | LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  exact probReal_textbook_regret_bound_deterministic_ge_of_confidenceEventUpTo_ge
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2
    (probReal_confidenceEventUpTo_ge_of_selfNormalizedConfidenceEventUpTo_ge
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      (P := P) h_self_prob)

lemma probReal_textbook_regret_bound_deterministic_ge_of_centeredNoiseBiasConfidenceEventUpTo_ge
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    {δ : ℝ}
    (h_noise_prob :
      1 - δ ≤
        P.real {ω | LinUCBCenteredNoiseBiasConfidenceEventUpTo A R reg β x ν θ n ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  exact probReal_textbook_regret_bound_deterministic_ge_of_selfNormalizedConfidenceEventUpTo_ge
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2
    (probReal_selfNormalizedConfidenceEventUpTo_ge_of_centeredNoiseBiasConfidenceEventUpTo_ge
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      (P := P) θ h_linear hreg_pos h_noise_prob)

/-- High-probability deterministic LinUCB regret bound, consuming the textbook determinant-ratio
self-normalized centered-noise event directly.

This is the endpoint expected from a future Gaussian-mixture concentration theorem: once that
theorem proves `LinUCBTextbookSelfNormalizedNoiseEventUpTo` with probability at least `1 - δ`, and
the determinant-ratio radius plus ridge-bias radius is dominated by `β`, the deterministic
textbook-shaped regret bound follows. -/
lemma probReal_textbook_regret_bound_deterministic_ge_of_textbookNoiseUpTo_ge
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (S2 : ℝ)
    (hθ : ParameterSqNormBound θ S2)
    {σ2 : ℝ≥0} {δ : ℝ}
    (h_budget : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      (√(textbookSelfNormalizedNoiseBound σ2 δ (designDetRatio A reg x t ω)) +
        √(reg * S2)) ^ 2 ≤ β (t + 1))
    (h_noise_prob :
      1 - δ ≤
        P.real {ω |
          LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  exact probReal_textbook_regret_bound_deterministic_ge_of_centeredNoiseBiasConfidenceEventUpTo_ge
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear
    (probReal_centeredNoiseBiasUpTo_ge_of_textbookNoiseUpTo_ge (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (P := P)
      θ S2 hreg_pos hθ h_budget h_noise_prob)


end LinUCB

end Bandits
