/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: OpenAI, Fawad Haider
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.LinUCB.Regret.Deterministic

/-!
# LinUCB Regret: Log-Determinant Bounds

This module contains the log-determinant regret interfaces. It sits between the deterministic
capped-width regret proof and the later finite-action textbook bounds.
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

/-- Textbook log-determinant finite-action LinUCB regret theorem on the horizon-local confidence
event. -/
lemma regret_ae_imp_le_textbook_logDet_finite_action_upTo
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg) :
    ∀ᵐ ω ∂P,
      LinUCBConfidenceEventUpTo A R reg β x ν n ω →
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) *
              √(ellipticalPotential A reg x n ω)) := by
  have h_gap_two : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 → gap ν (A t ω) ≤ 2 := by
    filter_upwards [gap_ae_le_of_GapBound (A := A) (ν := ν) (n := n) (P := P)
      2 (gapBound_two_of_meanRewardBound_neg_one_one (ν := ν) h_mean_bound)] with
      ω h_gapω t ht _ht0
    exact h_gapω t ht
  filter_upwards
    [regret_ae_imp_le_initial_gap_add_sqrt_nat_mul_beta_capped_sum_upTo
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      h h_gap_two hβ_schedule
      (widthQuadraticForm_ae_nonneg_of_reg_nonneg (A := A) (reg := reg)
        (x := x) (n := n) (P := P) hreg_pos.le),
     cappedQuadraticWidthSum_ae_le_ellipticalPotential_of_reg_pos
      (A := A) (reg := reg) (x := x) (n := n) (P := P) hreg_pos] with
      ω h_regret h_capped h_confω
  refine (h_regret h_confω).trans ?_
  gcongr

/-- Textbook log-determinant finite-action LinUCB regret theorem on the global confidence event. -/
lemma regret_ae_imp_le_textbook_logDet_finite_action
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg) :
    ∀ᵐ ω ∂P,
      LinUCBConfidenceEvent A R reg β x ν ω →
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) *
              √(ellipticalPotential A reg x n ω)) := by
  filter_upwards [regret_ae_imp_le_textbook_logDet_finite_action_upTo
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
    h h_mean_bound hβ_schedule hreg_pos] with ω h_regret h_confω
  exact h_regret h_confω.toUpTo

/-- The horizon-local confidence event is almost surely contained in the log-determinant regret
event. -/
lemma probReal_confidenceEventUpTo_le_textbook_logDet_regret_bound
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg) :
    P.real {ω | LinUCBConfidenceEventUpTo A R reg β x ν n ω} ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) * √(ellipticalPotential A reg x n ω))} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  filter_upwards [regret_ae_imp_le_textbook_logDet_finite_action_upTo
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
    h h_mean_bound hβ_schedule hreg_pos] with ω h_regret h_confω
  exact h_regret h_confω

/-- High-probability wrapper for the log-determinant regret event. -/
lemma probReal_textbook_logDet_regret_bound_ge_of_confidenceEventUpTo_ge
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg) {δ : ℝ}
    (h_conf_prob : 1 - δ ≤ P.real {ω | LinUCBConfidenceEventUpTo A R reg β x ν n ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) * √(ellipticalPotential A reg x n ω))} :=
  h_conf_prob.trans
    (probReal_confidenceEventUpTo_le_textbook_logDet_regret_bound
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      h h_mean_bound hβ_schedule hreg_pos)

/-- High-probability log-determinant regret wrapper from self-normalized confidence. -/
lemma probReal_textbook_logDet_regret_bound_ge_of_selfNormalizedConfidenceEventUpTo_ge
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg) {δ : ℝ}
    (h_self_prob :
      1 - δ ≤ P.real {ω | LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) * √(ellipticalPotential A reg x n ω))} := by
  exact probReal_textbook_logDet_regret_bound_ge_of_confidenceEventUpTo_ge
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
    h h_mean_bound hβ_schedule hreg_pos
    (probReal_confidenceEventUpTo_ge_of_selfNormalizedConfidenceEventUpTo_ge
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      (P := P) h_self_prob)

/-- Failure-probability wrapper for the log-determinant regret event. -/
lemma probReal_textbook_logDet_regret_bound_failure_le_of_confidenceEventUpTo_failure_le
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg) {δ : ℝ}
    (h_conf_failure :
      P.real {ω | ¬ LinUCBConfidenceEventUpTo A R reg β x ν n ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) * √(ellipticalPotential A reg x n ω))} ≤ δ := by
  exact (probReal_failure_le_of_ae_imp (P := P)
    (regret_ae_imp_le_textbook_logDet_finite_action_upTo (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      h h_mean_bound hβ_schedule hreg_pos)).trans h_conf_failure

/-- Failure-probability log-determinant regret wrapper from self-normalized confidence. -/
lemma probReal_textbook_logDet_regret_bound_failure_le_of_selfNormalizedUpTo_failure_le
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg) {δ : ℝ}
    (h_self_failure :
      P.real {ω | ¬ LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) * √(ellipticalPotential A reg x n ω))} ≤ δ := by
  exact probReal_textbook_logDet_regret_bound_failure_le_of_confidenceEventUpTo_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
    h h_mean_bound hβ_schedule hreg_pos
    (probReal_confidenceEventUpTo_failure_le_of_selfNormalizedConfidenceEventUpTo_failure_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      (P := P) h_self_failure)

/-- Failure-probability log-determinant regret bound from coordinate tail bounds. -/
lemma probReal_textbook_logDet_regret_bound_failure_le_of_coordTail_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0}
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (S2 : ℝ)
    (hθ : ParameterSqNormBound θ S2)
    {coordBudget : ℕ → ℝ} {δ : ℝ}
    (hcoord_nonneg : ∀ t, t ∈ range n → t ≠ 0 → 0 ≤ coordBudget (t + 1))
    (hβ_budget : ∀ t, t ∈ range n → t ≠ 0 →
      centeredNoiseCoordinateBetaBudget d reg S2 coordBudget (t + 1) ≤ β (t + 1))
    (hδ_nonneg : 0 ≤ δ) (hn : n ≠ 0) (hd : d ≠ 0)
    (h_tail : ∀ t, t ∈ range n → t ≠ 0 →
      centeredNoiseCoordinateTailBound σ2 L2 t (coordBudget (t + 1)) ≤ δ / (2 * n * d)) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) * √(ellipticalPotential A reg x n ω))} ≤ δ := by
  exact probReal_textbook_logDet_regret_bound_failure_le_of_confidenceEventUpTo_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
    h h_mean_bound hβ_schedule hreg_pos
    (probReal_confidenceEventUpTo_failure_le_of_coord_tail_le (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      h hν hreg_pos L2 hL2 θ h_linear S2 hθ hcoord_nonneg hβ_budget hδ_nonneg hn hd
      h_tail)

/-- High-probability log-determinant regret bound from coordinate tail bounds. -/
lemma probReal_textbook_logDet_regret_bound_ge_of_coordTail_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0}
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (S2 : ℝ)
    (hθ : ParameterSqNormBound θ S2)
    {coordBudget : ℕ → ℝ} {δ : ℝ}
    (hcoord_nonneg : ∀ t, t ∈ range n → t ≠ 0 → 0 ≤ coordBudget (t + 1))
    (hβ_budget : ∀ t, t ∈ range n → t ≠ 0 →
      centeredNoiseCoordinateBetaBudget d reg S2 coordBudget (t + 1) ≤ β (t + 1))
    (hδ_nonneg : 0 ≤ δ) (hn : n ≠ 0) (hd : d ≠ 0)
    (h_tail : ∀ t, t ∈ range n → t ≠ 0 →
      centeredNoiseCoordinateTailBound σ2 L2 t (coordBudget (t + 1)) ≤ δ / (2 * n * d)) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) * √(ellipticalPotential A reg x n ω))} := by
  exact probReal_textbook_logDet_regret_bound_ge_of_confidenceEventUpTo_ge
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
    h h_mean_bound hβ_schedule hreg_pos
    (probReal_confidenceEventUpTo_ge_of_coord_tail_le (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      h hν hreg_pos L2 hL2 θ h_linear S2 hθ hcoord_nonneg hβ_budget hδ_nonneg hn hd
      h_tail)

/-- Failure-probability log-determinant regret bound for the canonical coordinate log budget. -/
lemma probReal_textbook_logDet_regret_bound_failure_le_of_coordLogBudget
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0}
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (S2 : ℝ)
    (hθ : ParameterSqNormBound θ S2)
    {δ : ℝ}
    (hσ2_pos : 0 < (σ2 : ℝ)) (hL2_pos : 0 < L2) (hδ_pos : 0 < δ)
    (hn : n ≠ 0) (hd : d ≠ 0)
    (hβ_budget : ∀ t, t ∈ range n → t ≠ 0 →
      centeredNoiseCoordinateBetaBudget d reg S2
          (centeredNoiseCoordinateLogBudget σ2 L2 n d δ) (t + 1) ≤ β (t + 1)) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) * √(ellipticalPotential A reg x n ω))} ≤ δ := by
  exact probReal_textbook_logDet_regret_bound_failure_le_of_coordTail_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
    h hν h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear S2 hθ
    (fun t _ht _ht0 ↦ centeredNoiseCoordinateLogBudget_nonneg σ2 L2 n d δ (t + 1))
    hβ_budget hδ_pos.le hn hd
    (fun t _ht ht0 ↦ centeredNoiseCoordinateTailBound_logBudget_le (σ2 := σ2)
      (L2 := L2) (δ := δ) (n := n) (d := d) (t := t)
      hσ2_pos hL2_pos hδ_pos hn hd ht0)

/-- High-probability log-determinant regret bound for the canonical coordinate log budget. -/
lemma probReal_textbook_logDet_regret_bound_ge_of_coordLogBudget
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0}
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (S2 : ℝ)
    (hθ : ParameterSqNormBound θ S2)
    {δ : ℝ}
    (hσ2_pos : 0 < (σ2 : ℝ)) (hL2_pos : 0 < L2) (hδ_pos : 0 < δ)
    (hn : n ≠ 0) (hd : d ≠ 0)
    (hβ_budget : ∀ t, t ∈ range n → t ≠ 0 →
      centeredNoiseCoordinateBetaBudget d reg S2
          (centeredNoiseCoordinateLogBudget σ2 L2 n d δ) (t + 1) ≤ β (t + 1)) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * β n) * √(ellipticalPotential A reg x n ω))} := by
  exact probReal_event_ge_of_failure_le (P := P)
    (probReal_textbook_logDet_regret_bound_failure_le_of_coordLogBudget
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      h hν h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear S2 hθ hσ2_pos
      hL2_pos hδ_pos hn hd hβ_budget)

/-- Failure-probability log-determinant regret bound for the raw coordinate beta. -/
lemma probReal_textbook_logDet_regret_bound_failure_le_of_coordinateBeta
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0} (L2 S2 : ℝ) {δ : ℝ}
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (coordinateLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule (coordinateLinUCBBeta d reg S2 σ2 L2 n δ))
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hL2_pos : 0 < L2) (hδ_pos : 0 < δ)
    (hn : n ≠ 0) (hd : d ≠ 0) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * coordinateLinUCBBeta d reg S2 σ2 L2 n δ n) *
              √(ellipticalPotential A reg x n ω))} ≤ δ := by
  exact probReal_textbook_logDet_regret_bound_failure_le_of_coordLogBudget
    (A := A) (R := R) (reg := reg)
    (β := coordinateLinUCBBeta d reg S2 σ2 L2 n δ) (x := x) (ν := ν) (n := n)
    h hν h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear S2 hθ hσ2_pos
    hL2_pos hδ_pos hn hd (fun t _ht _ht0 ↦ le_rfl)

/-- High-probability log-determinant regret bound for the raw coordinate beta. -/
lemma probReal_textbook_logDet_regret_bound_ge_of_coordinateBeta
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0} (L2 S2 : ℝ) {δ : ℝ}
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (coordinateLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule (coordinateLinUCBBeta d reg S2 σ2 L2 n δ))
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hL2_pos : 0 < L2) (hδ_pos : 0 < δ)
    (hn : n ≠ 0) (hd : d ≠ 0) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * coordinateLinUCBBeta d reg S2 σ2 L2 n δ n) *
              √(ellipticalPotential A reg x n ω))} := by
  exact probReal_event_ge_of_failure_le (P := P)
    (probReal_textbook_logDet_regret_bound_failure_le_of_coordinateBeta
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n)
      L2 S2 h hν h_mean_bound hβ_schedule hreg_pos hL2 θ h_linear hθ hσ2_pos
      hL2_pos hδ_pos hn hd)

/-- Failure-probability log-determinant regret bound for the monotone coordinate beta. -/
lemma probReal_textbook_logDet_regret_bound_failure_le_of_coordinateMonotoneBeta
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0} (L2 S2 : ℝ) {δ : ℝ}
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hL2_pos : 0 < L2) (hδ_pos : 0 < δ)
    (hn : n ≠ 0) (hd : d ≠ 0) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ n) *
              √(ellipticalPotential A reg x n ω))} ≤ δ := by
  exact probReal_textbook_logDet_regret_bound_failure_le_of_coordLogBudget
    (A := A) (R := R) (reg := reg)
    (β := coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) (x := x) (ν := ν) (n := n)
    h hν h_mean_bound (coordinateLinUCBMonotoneBeta_schedule d reg S2 σ2 L2 n δ)
    hreg_pos L2 hL2 θ h_linear S2 hθ hσ2_pos hL2_pos hδ_pos hn hd
    (fun t _ht _ht0 ↦ coordinateLinUCBMonotoneBeta_dominates d reg S2 σ2 L2 n δ (t + 1))

/-- High-probability log-determinant regret bound for the monotone coordinate beta. -/
lemma probReal_textbook_logDet_regret_bound_ge_of_coordinateMonotoneBeta
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0} (L2 S2 : ℝ) {δ : ℝ}
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hL2_pos : 0 < L2) (hδ_pos : 0 < δ)
    (hn : n ≠ 0) (hd : d ≠ 0) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ n) *
              √(ellipticalPotential A reg x n ω))} := by
  exact probReal_event_ge_of_failure_le (P := P)
    (probReal_textbook_logDet_regret_bound_failure_le_of_coordinateMonotoneBeta
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n)
      L2 S2 h hν h_mean_bound hreg_pos hL2 θ h_linear hθ hσ2_pos hL2_pos hδ_pos
      hn hd)

/-- Failure-probability log-determinant regret bound for the monotone coordinate beta, including
the degenerate cases handled by the self-normalized confidence theorem. -/
lemma probReal_textbook_logDet_regret_bound_failure_le_of_coordinateMonotoneBeta_all
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0} (L2 S2 : ℝ) {δ : ℝ}
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ n) *
              √(ellipticalPotential A reg x n ω))} ≤ δ := by
  exact probReal_textbook_logDet_regret_bound_failure_le_of_selfNormalizedUpTo_failure_le
    (A := A) (R := R) (reg := reg)
    (β := coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) (x := x) (ν := ν) (n := n)
    h h_mean_bound (coordinateLinUCBMonotoneBeta_schedule d reg S2 σ2 L2 n δ)
    hreg_pos
    (probReal_selfNormalizedUpTo_failure_le_of_coordinateMonotoneBeta_all
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n)
      L2 S2 h hν hreg_pos hL2 θ h_linear hθ hσ2_pos hδ_pos)

/-- High-probability log-determinant regret bound for the monotone coordinate beta, including
the degenerate cases handled by the self-normalized confidence theorem. -/
lemma probReal_textbook_logDet_regret_bound_ge_of_coordinateMonotoneBeta_all
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0} (L2 S2 : ℝ) {δ : ℝ}
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else gap ν (A 0 ω)) +
            2 * (√((n : ℝ) * coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ n) *
              √(ellipticalPotential A reg x n ω))} := by
  exact probReal_textbook_logDet_regret_bound_ge_of_selfNormalizedConfidenceEventUpTo_ge
    (A := A) (R := R) (reg := reg)
    (β := coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) (x := x) (ν := ν) (n := n)
    h h_mean_bound (coordinateLinUCBMonotoneBeta_schedule d reg S2 σ2 L2 n δ)
    hreg_pos
    (probReal_selfNormalizedUpTo_ge_of_coordinateMonotoneBeta_all
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n)
      L2 S2 h hν hreg_pos hL2 θ h_linear hθ hσ2_pos hδ_pos)

end LinUCB

end Bandits
