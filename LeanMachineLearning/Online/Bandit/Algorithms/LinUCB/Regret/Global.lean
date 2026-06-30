/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: OpenAI, Fawad Haider
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.LinUCB.Regret.CoordinateTail

/-!
# LinUCB Regret: Global Compatibility Wrappers

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

/-- The confidence event is almost surely contained in the deterministic textbook regret-bound
event. This is the version to combine with a future high-probability confidence theorem. -/
lemma probReal_confidenceEvent_le_textbook_regret_bound_deterministic
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) :
    P.real {ω | LinUCBConfidenceEvent A R reg β x ν ω} ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  filter_upwards [regret_ae_imp_le_textbook_finite_action_deterministic_bound
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2] with ω h_regret h_confω
  exact h_regret h_confω

/-- High-probability wrapper for the deterministic textbook finite-action LinUCB regret bound. -/
lemma probReal_textbook_regret_bound_deterministic_ge_of_confidenceEvent_ge
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) {δ : ℝ}
    (h_conf_prob : 1 - δ ≤ P.real {ω | LinUCBConfidenceEvent A R reg β x ν ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  exact h_conf_prob.trans
    (probReal_confidenceEvent_le_textbook_regret_bound_deterministic (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_mean_bound hβ_schedule
      hreg_pos L2 hL2)

/-- High-probability deterministic LinUCB regret bound, consuming the self-normalized
prediction-error event directly.

This is the interface expected from the eventual self-normalized concentration theorem: once that
theorem proves `LinUCBSelfNormalizedConfidenceEvent` with probability at least `1 - δ`, the
deterministic textbook regret bound follows immediately. -/
lemma probReal_textbook_regret_bound_deterministic_ge_of_selfNormalizedConfidenceEvent_ge
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) {δ : ℝ}
    (h_self_prob :
      1 - δ ≤ P.real {ω | LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  exact probReal_textbook_regret_bound_deterministic_ge_of_confidenceEvent_ge
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2
    (probReal_confidenceEvent_ge_of_selfNormalizedConfidenceEvent_ge (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (P := P) h_self_prob)

/-- High-probability deterministic LinUCB regret bound, consuming the textbook parameter
ellipsoid confidence event directly.

This is the bridge closest to the textbook proof shape: a future self-normalized concentration
theorem can prove that the true parameter lies in all confidence ellipsoids with probability at
least `1 - δ`, and this lemma converts that event into the deterministic finite-action regret
bound. -/
lemma probReal_textbook_regret_bound_deterministic_ge_of_parameterEllipsoidConfidenceEvent_ge
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    {δ : ℝ}
    (h_ellipsoid_prob :
      1 - δ ≤ P.real {ω | LinUCBParameterEllipsoidConfidenceEvent A R reg β x θ ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  exact probReal_textbook_regret_bound_deterministic_ge_of_selfNormalizedConfidenceEvent_ge
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2
    (probReal_selfNormalizedConfidenceEvent_ge_of_parameterEllipsoidConfidenceEvent_ge
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (P := P)
      θ h_linear hreg_pos h_ellipsoid_prob)

/-- High-probability deterministic LinUCB regret bound, consuming the centered-noise-plus-bias
confidence event exposed by the least-squares decomposition.

This is the closest current interface to the textbook self-normalized proof: a future
self-normalized concentration theorem should prove the displayed centered-noise-plus-bias event for
the textbook choice of `β`, and this lemma immediately converts it to the finite-action regret
bound. -/
lemma probReal_textbook_regret_bound_deterministic_ge_of_centeredNoiseBiasConfidenceEvent_ge
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
      1 - δ ≤ P.real {ω | LinUCBCenteredNoiseBiasConfidenceEvent A R reg β x ν θ ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  exact probReal_textbook_regret_bound_deterministic_ge_of_parameterEllipsoidConfidenceEvent_ge
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear
    (probReal_parameterEllipsoidConfidenceEvent_ge_of_centeredNoiseBiasConfidenceEvent_ge
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (P := P)
      θ h_linear hreg_pos h_noise_prob)

/-- Failure-probability wrapper for the deterministic textbook finite-action LinUCB regret bound. -/
lemma probReal_textbook_regret_bound_deterministic_failure_le_of_confidenceEvent_failure_le
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) {δ : ℝ}
    (h_conf_failure :
      P.real {ω | ¬ LinUCBConfidenceEvent A R reg β x ν ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ := by
  refine le_trans ?_ h_conf_failure
  refine probReal_failure_le_of_ae_imp (P := P) ?_
  filter_upwards [regret_ae_imp_le_textbook_finite_action_deterministic_bound
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2] with ω h_regret h_confω
  exact h_regret h_confω

/-- Failure-probability deterministic LinUCB regret bound, consuming the self-normalized
prediction-error event directly. -/
lemma probReal_textbook_regret_bound_deterministic_failure_le_of_selfNormalized_failure_le
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) {δ : ℝ}
    (h_self_failure :
      P.real {ω | ¬ LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ := by
  exact probReal_textbook_regret_bound_deterministic_failure_le_of_confidenceEvent_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2
    (probReal_confidenceEvent_failure_le_of_selfNormalizedConfidenceEvent_failure_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (P := P)
      h_self_failure)

/-- Failure-probability deterministic LinUCB regret bound, consuming the textbook parameter
ellipsoid confidence event directly. -/
lemma probReal_textbook_regret_bound_deterministic_failure_le_of_parameterEllipsoid_failure_le
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    {δ : ℝ}
    (h_ellipsoid_failure :
      P.real {ω | ¬ LinUCBParameterEllipsoidConfidenceEvent A R reg β x θ ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ := by
  exact probReal_textbook_regret_bound_deterministic_failure_le_of_selfNormalized_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2
    (probReal_selfNormalized_failure_le_of_parameterEllipsoid_failure_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (P := P)
      θ h_linear hreg_pos h_ellipsoid_failure)

/-- Failure-probability deterministic LinUCB regret bound, consuming the centered-noise-plus-bias
confidence event exposed by the least-squares decomposition. -/
lemma probReal_textbook_regret_bound_deterministic_failure_le_of_centeredNoiseBias_failure_le
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    {δ : ℝ}
    (h_noise_failure :
      P.real {ω | ¬ LinUCBCenteredNoiseBiasConfidenceEvent A R reg β x ν θ ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ := by
  exact probReal_textbook_regret_bound_deterministic_failure_le_of_parameterEllipsoid_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear
    (probReal_parameterEllipsoid_failure_le_of_centeredNoiseBias_failure_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (P := P)
      θ h_linear hreg_pos h_noise_failure)

/-- The confidence event is almost surely contained in the textbook finite-action regret-bound
event.

This is the probability bridge needed after the good-event theorem: once a concentration theorem
proves that `LinUCBConfidenceEvent` has high probability, this lemma transfers that probability
mass to the displayed regret bound. -/
lemma probReal_confidenceEvent_le_textbook_regret_bound
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) :
    P.real {ω | LinUCBConfidenceEvent A R reg β x ν ω} ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) *
              √(2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ)))))} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  filter_upwards [regret_ae_imp_le_textbook_finite_action (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_mean_bound hβ_schedule
    hreg_pos L2 hL2] with ω h_regret h_confω
  exact h_regret h_confω

/-- High-probability wrapper for the textbook finite-action LinUCB regret bound.

If a future self-normalized concentration theorem proves that the confidence event has probability
at least `1 - δ`, then the textbook regret bound has probability at least `1 - δ` as well. -/
lemma probReal_textbook_regret_bound_ge_of_confidenceEvent_ge
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) {δ : ℝ}
    (h_conf_prob : 1 - δ ≤ P.real {ω | LinUCBConfidenceEvent A R reg β x ν ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) *
              √(2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ)))))} := by
  exact h_conf_prob.trans
    (probReal_confidenceEvent_le_textbook_regret_bound (A := A) (R := R) (reg := reg)
      (β := β) (x := x) (ν := ν) (n := n) h h_mean_bound hβ_schedule hreg_pos L2 hL2)

/-- Failure-probability wrapper for the textbook finite-action LinUCB regret bound.

If a future self-normalized concentration theorem proves that the confidence event fails with
probability at most `δ`, then the textbook regret bound fails with probability at most `δ`. -/
lemma probReal_textbook_regret_bound_failure_le_of_confidenceEvent_failure_le
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) {δ : ℝ}
    (h_conf_failure :
      P.real {ω | ¬ LinUCBConfidenceEvent A R reg β x ν ω} ≤ δ) :
    P.real {ω |
      ¬
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) *
              √(2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ)))))} ≤ δ := by
  refine le_trans ?_ h_conf_failure
  refine probReal_failure_le_of_ae_imp (P := P) ?_
  filter_upwards [regret_ae_imp_le_textbook_finite_action (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_mean_bound hβ_schedule
    hreg_pos L2 hL2] with ω h_regret h_confω
  exact h_regret h_confω

/-- Corollary of `regret_ae_imp_le_textbook_finite_action` when the confidence event is known to
hold almost surely. This is stronger than the textbook high-probability route and is mainly useful
as a compatibility wrapper for earlier lemmas in this file. -/
lemma regret_ae_le_textbook_finite_action
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_conf : ∀ᵐ ω ∂P, LinUCBConfidenceEvent A R reg β x ν ω)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) :
    ∀ᵐ ω ∂P,
      regret ν A n ω ≤
        (if n = 0 then 0 else gap ν (A 0 ω)) +
          2 * (√((n : ℝ) * β n) *
            √(2 * (d : ℝ) * Real.log (1 + (n : ℝ) * L2 / (reg * (d : ℝ))))) := by
  filter_upwards [regret_ae_imp_le_textbook_finite_action (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_mean_bound hβ_schedule
    hreg_pos L2 hL2, h_conf] with ω h_regret h_confω
  exact h_regret h_confω

/-- Almost surely, cumulative regret is bounded by the simplified initial-gap term plus
`2 * √(n * β n) * √W` whenever positive regularization, the positive-time width cap, and the final
log-determinant potential bound hold.

The capped-sum/log-determinant part of the elliptical-potential argument is proved internally:
positive regularization gives determinant nonvanishing and nonnegative quadratic forms, while
`h_quad_le_one` lets this older theorem feed the uncapped `widthSqSum` regret route. -/
lemma regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_of_ellipticalPotential_bound
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_best : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) n ω)
    (h_arm : ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      estimatedReward A R reg x (A n ω) n ω -
        √(β (n + 1)) * width A reg x (A n ω) n ω ≤ (ν (A n ω))[id])
    (hβ : ∀ t, 0 ≤ β (t + 1)) (hβ_mono : Monotone β) (W : ℝ)
    (hreg_pos : 0 < reg)
    (h_quad_le_one : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      widthQuadraticForm A reg x (A t ω) t ω ≤ 1)
    (h_potential_le : ∀ᵐ ω ∂P, ellipticalPotential A reg x n ω ≤ W) :
    ∀ᵐ ω ∂P,
      regret ν A n ω ≤
        (if n = 0 then 0 else gap ν (A 0 ω)) + 2 * (√((n : ℝ) * β n) * √W) := by
  exact regret_ae_le_initial_gap_add_sqrt_nat_mul_beta_capped_quadratic_width_bound
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h h_best
    h_arm hβ hβ_mono W
    (cappedQuadraticWidthBound_ae_of_reg_pos_det_update_ellipticalPotential_le_bound
      (A := A) (reg := reg) (x := x) (n := n) (P := P) (W := W) hreg_pos
      h_quad_le_one h_potential_le)


end LinUCB

end Bandits
