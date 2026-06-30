/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: OpenAI, Fawad Haider
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.LinUCB.Regret.Degenerate

/-!
# LinUCB Regret: Coordinate Confidence

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

/-- Conservative finite-action self-normalized confidence theorem for the monotone
coordinate-union beta schedule.

This is not the sharp vector self-normalized theorem from the textbook. It is the proved
finite-coordinate substitute: scalar subgaussian concentration is applied to each coordinate and
union-bounded over coordinates and positive times, then the deterministic least-squares bridges
turn the coordinate event into the usual LinUCB prediction-error event. -/
lemma probReal_selfNormalizedUpTo_failure_le_of_coordinateMonotoneBeta
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
    (hσ2_pos : 0 < (σ2 : ℝ)) (hL2_pos : 0 < L2) (hδ_pos : 0 < δ)
    (hn : n ≠ 0) (hd : d ≠ 0) :
    P.real {ω |
      ¬ LinUCBSelfNormalizedConfidenceEventUpTo A R reg
          (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) x ν n ω} ≤ δ := by
  refine probReal_selfNormalizedUpTo_failure_le_of_coord_tail_le
    (A := A) (R := R) (reg := reg)
    (β := coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ)
    (x := x) (ν := ν) (n := n)
    (coordBudget := centeredNoiseCoordinateLogBudget σ2 L2 n d δ)
    h hν hreg_pos L2 hL2 θ h_linear S2 hθ ?_ ?_ hδ_pos.le hn hd ?_
  · intro t _ht _ht0
    exact centeredNoiseCoordinateLogBudget_nonneg σ2 L2 n d δ (t + 1)
  · intro t _ht _ht0
    exact coordinateLinUCBMonotoneBeta_dominates d reg S2 σ2 L2 n δ (t + 1)
  · intro t _ht ht0
    exact centeredNoiseCoordinateTailBound_logBudget_le (σ2 := σ2) (L2 := L2)
      (δ := δ) (n := n) (d := d) (t := t) hσ2_pos hL2_pos hδ_pos hn hd ht0

/-- Conservative finite-action self-normalized confidence theorem for the monotone
coordinate-union beta schedule, including zero-horizon, zero-dimensional, and nonpositive-feature
bound cases. -/
lemma probReal_selfNormalizedUpTo_failure_le_of_coordinateMonotoneBeta_all
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
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) :
    P.real {ω |
      ¬ LinUCBSelfNormalizedConfidenceEventUpTo A R reg
          (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) x ν n ω} ≤ δ := by
  by_cases hL2_pos : 0 < L2
  · by_cases hn : n = 0
    · exact probReal_failure_le_of_forall (P := P) (fun ω ↦
        selfNormalizedConfidenceEventUpTo_of_zero_horizon (A := A) (R := R)
          (reg := reg) (x := x) (ν := ν) (n := n) (ω := ω)
          (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) hn) hδ_pos.le
    · by_cases hd : d = 0
      · exact probReal_failure_le_of_forall (P := P) (fun ω ↦
          selfNormalizedConfidenceEventUpTo_of_linear_dim_eq_zero (A := A) (R := R)
            (reg := reg) (x := x) (ν := ν) (n := n) (ω := ω)
            (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) hd h_linear) hδ_pos.le
      · exact
          probReal_selfNormalizedUpTo_failure_le_of_coordinateMonotoneBeta
            (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n)
            L2 S2 h hν hreg_pos hL2 θ h_linear hθ hσ2_pos hL2_pos
            hδ_pos hn hd
  · have hL2_nonpos : L2 ≤ 0 := le_of_not_gt hL2_pos
    exact probReal_failure_le_of_forall (P := P) (fun ω ↦
      selfNormalizedConfidenceEventUpTo_of_featureSqNormBound_nonpos (A := A) (R := R)
        (reg := reg) (x := x) (ν := ν) (n := n) (ω := ω)
        (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) hL2 hL2_nonpos h_linear)
      hδ_pos.le

/-- High-probability form of the conservative finite-action self-normalized confidence theorem for
the monotone coordinate-union beta schedule. -/
lemma probReal_selfNormalizedUpTo_ge_of_coordinateMonotoneBeta_all
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
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) :
    1 - δ ≤
      P.real {ω |
        LinUCBSelfNormalizedConfidenceEventUpTo A R reg
          (coordinateLinUCBMonotoneBeta d reg S2 σ2 L2 n δ) x ν n ω} := by
  exact probReal_event_ge_of_failure_le (P := P)
    (probReal_selfNormalizedUpTo_failure_le_of_coordinateMonotoneBeta_all
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n)
      L2 S2 h hν hreg_pos hL2 θ h_linear hθ hσ2_pos hδ_pos)


end LinUCB

end Bandits
