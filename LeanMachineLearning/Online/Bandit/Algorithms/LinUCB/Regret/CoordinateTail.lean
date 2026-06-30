/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: OpenAI, Fawad Haider
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.LinUCB.Regret.TextbookFailure

/-!
# LinUCB Regret: Coordinate Tail Route

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

/-- Failure-probability deterministic LinUCB regret bound, consuming the horizon-local
self-normalized centered-noise event plus the deterministic ridge-bias radius. -/
lemma probReal_textbookRegretDet_failure_le_of_centeredNoiseUpTo_failure_le
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
    {noiseBudget : ℕ → ℝ} {δ : ℝ}
    (h_budget : ∀ t, t ∈ range n → t ≠ 0 →
      (√(noiseBudget (t + 1)) + √(reg * S2)) ^ 2 ≤ β (t + 1))
    (h_noise_failure :
      P.real {ω | ¬ LinUCBCenteredNoiseConfidenceEventUpTo A R reg noiseBudget x ν n ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ := by
  exact probReal_textbook_regret_bound_deterministic_failure_le_of_centeredNoiseBiasUpTo_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear
    (probReal_centeredNoiseBiasUpTo_failure_le_of_centeredNoiseUpTo_failure_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      (P := P) θ S2 hreg_pos hθ h_budget h_noise_failure)

/-- Failure-probability deterministic LinUCB regret bound, consuming a coordinate-wise
finite-horizon centered-noise event.

This is not the final textbook concentration theorem. It is a conservative scalar-projection route
that can reuse coordinate tail bounds while the exact vector self-normalized theorem is being
formalized. -/
lemma probReal_textbookRegretDet_failure_le_of_coordinateBoundUpTo_failure_le
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
    {coordBudget noiseBudget : ℕ → ℝ} {δ : ℝ}
    (hcoord_nonneg : ∀ t, t ∈ range n → t ≠ 0 → 0 ≤ coordBudget (t + 1))
    (h_coord_budget : ∀ t, t ∈ range n → t ≠ 0 →
      (d : ℝ) * coordBudget (t + 1) ^ 2 / reg ≤ noiseBudget (t + 1))
    (h_noise_budget : ∀ t, t ∈ range n → t ≠ 0 →
      (√(noiseBudget (t + 1)) + √(reg * S2)) ^ 2 ≤ β (t + 1))
    (hcoord_failure :
      P.real {ω | ¬ LinUCBCenteredNoiseCoordinateBoundEventUpTo A R coordBudget x ν n ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ := by
  exact probReal_textbookRegretDet_failure_le_of_centeredNoiseUpTo_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear S2 hθ h_noise_budget
    (probReal_centeredNoiseConfidenceEventUpTo_failure_le_of_coordinateBoundEventUpTo_failure_le
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) (P := P)
      hreg_pos hcoord_nonneg h_coord_budget hcoord_failure)

/-- Failure-probability deterministic LinUCB regret bound derived from scalar coordinate
concentration and a finite union bound.

This theorem removes the probability assumption on the coordinate event in the conservative
coordinate route. The remaining hypotheses are deterministic radius/budget checks connecting the
coordinate budget to the centered-noise quadratic budget and then to the LinUCB `β` radius. -/
lemma probReal_textbookRegretDet_failure_le_coordinateTailSum
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (S2 : ℝ)
    (hθ : ParameterSqNormBound θ S2)
    {coordBudget noiseBudget : ℕ → ℝ}
    (hcoord_nonneg : ∀ t, t ∈ range n → t ≠ 0 → 0 ≤ coordBudget (t + 1))
    (h_coord_budget : ∀ t, t ∈ range n → t ≠ 0 →
      (d : ℝ) * coordBudget (t + 1) ^ 2 / reg ≤ noiseBudget (t + 1))
    (h_noise_budget : ∀ t, t ∈ range n → t ≠ 0 →
      (√(noiseBudget (t + 1)) + √(reg * S2)) ^ 2 ≤ β (t + 1)) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤
      centeredNoiseCoordinateTailFailureSum d σ2 L2 coordBudget n := by
  exact probReal_textbookRegretDet_failure_le_of_coordinateBoundUpTo_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear S2 hθ hcoord_nonneg
    h_coord_budget h_noise_budget
    (probReal_centeredNoiseCoordinateBoundEventUpTo_failure_le_tail_sum
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      (P := P) h hν L2 hL2 hcoord_nonneg)

/-- Failure-probability deterministic LinUCB regret bound from scalar coordinate concentration,
with the coordinate finite-union tail sum bounded by a user-supplied `δ`. -/
lemma probReal_textbookRegretDet_failure_le_of_coordinateTailSum_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (S2 : ℝ)
    (hθ : ParameterSqNormBound θ S2)
    {coordBudget noiseBudget : ℕ → ℝ} {δ : ℝ}
    (hcoord_nonneg : ∀ t, t ∈ range n → t ≠ 0 → 0 ≤ coordBudget (t + 1))
    (h_coord_budget : ∀ t, t ∈ range n → t ≠ 0 →
      (d : ℝ) * coordBudget (t + 1) ^ 2 / reg ≤ noiseBudget (t + 1))
    (h_noise_budget : ∀ t, t ∈ range n → t ≠ 0 →
      (√(noiseBudget (t + 1)) + √(reg * S2)) ^ 2 ≤ β (t + 1))
    (h_tail_sum : centeredNoiseCoordinateTailFailureSum d σ2 L2 coordBudget n ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ :=
  (probReal_textbookRegretDet_failure_le_coordinateTailSum
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h hν
    h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear S2 hθ hcoord_nonneg
    h_coord_budget h_noise_budget).trans h_tail_sum

/-- Failure-probability deterministic LinUCB regret bound from scalar coordinate concentration,
using the canonical coordinate-derived noise budget. -/
lemma probReal_textbookRegretDet_failure_le_coordTailSum_of_betaBudget
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (S2 : ℝ)
    (hθ : ParameterSqNormBound θ S2)
    {coordBudget : ℕ → ℝ}
    (hcoord_nonneg : ∀ t, t ∈ range n → t ≠ 0 → 0 ≤ coordBudget (t + 1))
    (hβ_budget : ∀ t, t ∈ range n → t ≠ 0 →
      centeredNoiseCoordinateBetaBudget d reg S2 coordBudget (t + 1) ≤ β (t + 1)) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤
      centeredNoiseCoordinateTailFailureSum d σ2 L2 coordBudget n := by
  refine probReal_textbookRegretDet_failure_le_coordinateTailSum
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
    (noiseBudget := centeredNoiseCoordinateNoiseBudget d reg coordBudget) h hν
    h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear S2 hθ hcoord_nonneg ?_ ?_
  · intro t _ht _ht0
    simp [centeredNoiseCoordinateNoiseBudget]
  · intro t ht ht0
    simpa [centeredNoiseCoordinateBetaBudget] using hβ_budget t ht ht0

/-- High-probability deterministic LinUCB regret bound from scalar coordinate concentration,
using the total coordinate finite-union tail sum as the failure budget. -/
lemma probReal_textbookRegretDet_ge_coordTailSum_of_betaBudget
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (S2 : ℝ)
    (hθ : ParameterSqNormBound θ S2)
    {coordBudget : ℕ → ℝ}
    (hcoord_nonneg : ∀ t, t ∈ range n → t ≠ 0 → 0 ≤ coordBudget (t + 1))
    (hβ_budget : ∀ t, t ∈ range n → t ≠ 0 →
      centeredNoiseCoordinateBetaBudget d reg S2 coordBudget (t + 1) ≤ β (t + 1)) :
    1 - centeredNoiseCoordinateTailFailureSum d σ2 L2 coordBudget n ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} :=
  probReal_event_ge_of_failure_le (P := P)
    (probReal_textbookRegretDet_failure_le_coordTailSum_of_betaBudget
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h hν
      h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear S2 hθ hcoord_nonneg hβ_budget)

/-- Failure-probability deterministic LinUCB regret bound from scalar coordinate concentration,
with the coordinate finite-union tail sum bounded by a user-supplied `δ` and the canonical
coordinate-derived noise budget. -/
lemma probReal_textbookRegretDet_failure_le_of_coordTailSum_le_of_betaBudget
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
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
    (h_tail_sum : centeredNoiseCoordinateTailFailureSum d σ2 L2 coordBudget n ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ :=
  (probReal_textbookRegretDet_failure_le_coordTailSum_of_betaBudget
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h hν
    h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear S2 hθ hcoord_nonneg
    hβ_budget).trans h_tail_sum

/-- High-probability deterministic LinUCB regret bound from scalar coordinate concentration,
when the total coordinate finite-union tail sum is bounded by a user-supplied `δ`. -/
lemma probReal_textbookRegretDet_ge_of_coordTailSum_le_of_betaBudget
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
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
    (h_tail_sum : centeredNoiseCoordinateTailFailureSum d σ2 L2 coordBudget n ≤ δ) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} :=
  probReal_event_ge_of_failure_le (P := P)
    (probReal_textbookRegretDet_failure_le_of_coordTailSum_le_of_betaBudget
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h hν
      h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear S2 hθ hcoord_nonneg
      hβ_budget h_tail_sum)

/-- Failure-probability deterministic LinUCB regret bound from scalar coordinate concentration,
where each positive time contributes at most `δ / n` to the finite-union coordinate-tail budget. -/
lemma probReal_textbookRegretDet_failure_le_of_coordTimeTail_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
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
    (hδ_nonneg : 0 ≤ δ)
    (h_tail : ∀ t, t ∈ range n → t ≠ 0 →
      (∑ _i : Fin d, 2 * centeredNoiseCoordinateTailBound σ2 L2 t
        (coordBudget (t + 1))) ≤ δ / (n : ℝ)) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ := by
  exact probReal_textbookRegretDet_failure_le_of_coordTailSum_le_of_betaBudget
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h hν
    h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear S2 hθ hcoord_nonneg
    hβ_budget
    (centeredNoiseCoordinateTailFailureSum_le_of_time_tail_le (d := d) (n := n)
      (σ2 := σ2) (L2 := L2) (coordBudget := coordBudget) hδ_nonneg h_tail)

/-- Failure-probability deterministic LinUCB regret bound from scalar coordinate concentration,
where every coordinate tail at each positive time is bounded by `δ / (2 n d)`. -/
lemma probReal_textbookRegretDet_failure_le_of_coordTail_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
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
      centeredNoiseCoordinateTailBound σ2 L2 t (coordBudget (t + 1)) ≤
        δ / (2 * (n : ℝ) * (d : ℝ))) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ := by
  exact probReal_textbook_regret_bound_deterministic_failure_le_of_confidenceEventUpTo_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2
    (probReal_confidenceEventUpTo_failure_le_of_coord_tail_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h hν
      hreg_pos L2 hL2 θ h_linear S2 hθ hcoord_nonneg hβ_budget hδ_nonneg hn hd h_tail)

/-- High-probability deterministic LinUCB regret bound from scalar coordinate concentration,
where every coordinate tail at each positive time is bounded by `δ / (2 n d)`.

This is the high-probability counterpart of
`probReal_textbookRegretDet_failure_le_of_coordTail_le`, stated in the direction closest to the
textbook theorem. -/
lemma probReal_textbookRegretDet_ge_of_coordTail_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
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
      centeredNoiseCoordinateTailBound σ2 L2 t (coordBudget (t + 1)) ≤
        δ / (2 * (n : ℝ) * (d : ℝ))) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  exact probReal_textbook_regret_bound_deterministic_ge_of_confidenceEventUpTo_ge
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2
    (probReal_confidenceEventUpTo_ge_of_coord_tail_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h hν
      hreg_pos L2 hL2 θ h_linear S2 hθ hcoord_nonneg hβ_budget hδ_nonneg hn hd h_tail)

/-- Failure-probability deterministic LinUCB regret bound using the explicit coordinate
log-budget. This is the deterministic-bound counterpart of
`probReal_textbook_logDet_regret_bound_failure_le_of_coordLogBudget`. -/
lemma probReal_textbookRegretDet_failure_le_of_coordLogBudget
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
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
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ := by
  refine probReal_textbookRegretDet_failure_le_of_coordTail_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
    (coordBudget := centeredNoiseCoordinateLogBudget σ2 L2 n d δ) h hν h_mean_bound
    hβ_schedule hreg_pos L2 hL2 θ h_linear S2 hθ ?_ hβ_budget hδ_pos.le hn hd ?_
  · intro t _ht _ht0
    exact centeredNoiseCoordinateLogBudget_nonneg σ2 L2 n d δ (t + 1)
  · intro t _ht ht0
    exact centeredNoiseCoordinateTailBound_logBudget_le (σ2 := σ2) (L2 := L2)
      (δ := δ) (n := n) (d := d) (t := t) hσ2_pos hL2_pos hδ_pos hn hd ht0

/-- High-probability deterministic LinUCB regret bound using the explicit coordinate
log-budget. -/
lemma probReal_textbookRegretDet_ge_of_coordLogBudget
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
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
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  exact probReal_event_ge_of_failure_le (P := P)
    (probReal_textbookRegretDet_failure_le_of_coordLogBudget
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h hν
      h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear S2 hθ hσ2_pos hL2_pos
      hδ_pos hn hd hβ_budget)

/-- Failure-probability deterministic LinUCB regret bound for the canonical conservative
coordinate-union beta schedule. -/
lemma probReal_textbookRegretDet_failure_le_of_coordinateBeta
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0}
    (L2 : ℝ) (S2 : ℝ) {δ : ℝ}
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
          (if n = 0 then 0 else 2) +
            textbookRegretBonus (d := d) reg
              (coordinateLinUCBBeta d reg S2 σ2 L2 n δ) L2 n} ≤ δ := by
  refine probReal_textbookRegretDet_failure_le_of_coordLogBudget
    (A := A) (R := R) (reg := reg)
    (β := coordinateLinUCBBeta d reg S2 σ2 L2 n δ) (x := x) (ν := ν)
    (n := n) h hν h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear S2
    hθ hσ2_pos hL2_pos hδ_pos hn hd ?_
  intro t _ht _ht0
  rfl

/-- High-probability deterministic LinUCB regret bound for the canonical conservative
coordinate-union beta schedule. -/
lemma probReal_textbookRegretDet_ge_of_coordinateBeta
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0}
    (L2 : ℝ) (S2 : ℝ) {δ : ℝ}
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
          (if n = 0 then 0 else 2) +
            textbookRegretBonus (d := d) reg
              (coordinateLinUCBBeta d reg S2 σ2 L2 n δ) L2 n} := by
  exact probReal_event_ge_of_failure_le (P := P)
    (probReal_textbookRegretDet_failure_le_of_coordinateBeta
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n)
      L2 S2 h hν h_mean_bound hβ_schedule hreg_pos hL2 θ h_linear hθ
      hσ2_pos hL2_pos hδ_pos hn hd)

/-- Failure-probability deterministic LinUCB regret bound for the monotone conservative
coordinate-union beta schedule. -/
lemma probReal_textbookRegretDet_failure_le_of_coordinateMonotoneBeta
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0}
    (L2 : ℝ) (S2 : ℝ) {δ : ℝ}
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
          (if n = 0 then 0 else 2) +
            textbookRegretBonus (d := d) reg
              (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) L2 n} ≤ δ := by
  refine probReal_textbookRegretDet_failure_le_of_coordLogBudget
    (A := A) (R := R) (reg := reg)
    (β := coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) (x := x) (ν := ν)
    (n := n) h hν h_mean_bound
    (coordinateLinUCBMonotoneBeta_schedule d reg S2 σ2 L2 n δ) hreg_pos L2 hL2 θ
    h_linear S2 hθ hσ2_pos hL2_pos hδ_pos hn hd ?_
  intro t _ht _ht0
  exact coordinateLinUCBMonotoneBeta_dominates d reg S2 σ2 L2 n δ (t + 1)

/-- High-probability deterministic LinUCB regret bound for the monotone conservative
coordinate-union beta schedule. -/
lemma probReal_textbookRegretDet_ge_of_coordinateMonotoneBeta
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0}
    (L2 : ℝ) (S2 : ℝ) {δ : ℝ}
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
          (if n = 0 then 0 else 2) +
            textbookRegretBonus (d := d) reg
              (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) L2 n} := by
  exact probReal_event_ge_of_failure_le (P := P)
    (probReal_textbookRegretDet_failure_le_of_coordinateMonotoneBeta
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n)
      L2 S2 h hν h_mean_bound hreg_pos hL2 θ h_linear hθ hσ2_pos hL2_pos
      hδ_pos hn hd)

/-- Failure-probability deterministic LinUCB regret bound for the monotone conservative
coordinate-union beta schedule, including the zero-horizon and zero-dimensional cases. -/
lemma probReal_textbookRegretDet_failure_le_of_coordinateMonotoneBeta_all
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0}
    (L2 : ℝ) (S2 : ℝ) {δ : ℝ}
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
          (if n = 0 then 0 else 2) +
            textbookRegretBonus (d := d) reg
              (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) L2 n} ≤ δ := by
  exact probReal_textbook_regret_bound_deterministic_failure_le_of_selfNormalizedUpTo_failure_le
    (A := A) (R := R) (reg := reg)
    (β := coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) (x := x) (ν := ν)
    (n := n) h h_mean_bound
    (coordinateLinUCBMonotoneBeta_schedule d reg S2 σ2 L2 n δ) hreg_pos L2 hL2
    (probReal_selfNormalizedUpTo_failure_le_of_coordinateMonotoneBeta_all
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n)
      L2 S2 h hν hreg_pos hL2 θ h_linear hθ hσ2_pos hδ_pos)

/-- Failure-probability deterministic LinUCB regret bound with the bounded-mean assumption
derived from linear realizability and normalized feature/parameter norm bounds. -/
lemma probReal_textbookRegretDet_failure_le_of_coordinateMonotoneBeta_all_of_linear_sq_norm_bounds
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0}
    (L2 : ℝ) (S2 : ℝ) {δ : ℝ}
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookRegretBonus (d := d) reg
              (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) L2 n} ≤ δ := by
  exact probReal_textbookRegretDet_failure_le_of_coordinateMonotoneBeta_all
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) L2 S2 h hν
    (meanRewardBound_neg_one_one_of_linear_sq_norm_bounds
      (ν := ν) (x := x) (θ := θ) h_linear hL2 hθ hLS_le_one)
    hreg_pos hL2 θ h_linear hθ hσ2_pos hδ_pos

/-- High-probability deterministic LinUCB regret bound for the monotone conservative
coordinate-union beta schedule, including the zero-horizon and zero-dimensional cases. -/
lemma probReal_textbookRegretDet_ge_of_coordinateMonotoneBeta_all
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0}
    (L2 : ℝ) (S2 : ℝ) {δ : ℝ}
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
          (if n = 0 then 0 else 2) +
            textbookRegretBonus (d := d) reg
              (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) L2 n} := by
  exact probReal_textbook_regret_bound_deterministic_ge_of_selfNormalizedConfidenceEventUpTo_ge
    (A := A) (R := R) (reg := reg)
    (β := coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) (x := x) (ν := ν)
    (n := n) h h_mean_bound
    (coordinateLinUCBMonotoneBeta_schedule d reg S2 σ2 L2 n δ) hreg_pos L2 hL2
    (probReal_selfNormalizedUpTo_ge_of_coordinateMonotoneBeta_all
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n)
      L2 S2 h hν hreg_pos hL2 θ h_linear hθ hσ2_pos hδ_pos)

/-- High-probability deterministic LinUCB regret bound with the bounded-mean assumption derived
from linear realizability and normalized feature/parameter norm bounds. -/
lemma probReal_textbookRegretDet_ge_of_coordinateMonotoneBeta_all_of_linear_sq_norm_bounds
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0}
    (L2 : ℝ) (S2 : ℝ) {δ : ℝ}
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookRegretBonus (d := d) reg
              (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) L2 n} := by
  exact probReal_textbookRegretDet_ge_of_coordinateMonotoneBeta_all
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) L2 S2 h hν
    (meanRewardBound_neg_one_one_of_linear_sq_norm_bounds
      (ν := ν) (x := x) (θ := θ) h_linear hL2 hθ hLS_le_one)
    hreg_pos hL2 θ h_linear hθ hσ2_pos hδ_pos


end LinUCB

end Bandits
