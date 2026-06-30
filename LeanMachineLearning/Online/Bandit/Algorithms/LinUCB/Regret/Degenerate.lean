/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: OpenAI, Fawad Haider
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.LinUCB.Regret.Core

/-!
# LinUCB Regret: Degenerate Cases

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

omit [IsMarkovKernel ν] in
/-- In zero feature dimension, the confidence event bounds cumulative regret by the initial gap.
There is no positive-time width contribution because all widths are zero. -/
lemma regret_le_initial_gap_of_confidence_dim_eq_zero [Nonempty (Fin K)]
    (hd : d = 0) (h_conf : LinUCBConfidenceEvent A R reg β x ν ω) :
    regret ν A n ω ≤ if n = 0 then 0 else gap ν (A 0 ω) := by
  refine (regret_le_sum_of_gap_bound (A := A) (ν := ν) (n := n) (ω := ω)
    (B := fun t ↦ if t = 0 then gap ν (A 0 ω) else 0) ?_).trans ?_
  · intro t _ht
    by_cases ht0 : t = 0
    · simp [ht0]
    · simpa [ht0] using
        gap_nonpos_of_confidence_dim_eq_zero (A := A) (R := R) (reg := reg)
          (β := β) (x := x) (ν := ν) (ω := ω) hd h_conf t ht0
  · rw [initial_gap_sum_eq]

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Horizon-local zero-dimensional regret bound. Only confidence inequalities up to the regret
horizon are needed. -/
lemma regret_le_initial_gap_of_confidenceUpTo_dim_eq_zero [Nonempty (Fin K)]
    (hd : d = 0) (h_conf : LinUCBConfidenceEventUpTo A R reg β x ν n ω) :
    regret ν A n ω ≤ if n = 0 then 0 else gap ν (A 0 ω) := by
  refine (regret_le_sum_of_gap_bound (A := A) (ν := ν) (n := n) (ω := ω)
    (B := fun t ↦ if t = 0 then gap ν (A 0 ω) else 0) ?_).trans ?_
  · intro t ht
    by_cases ht0 : t = 0
    · simp [ht0]
    · simpa [ht0] using
        gap_nonpos_of_confidenceUpTo_dim_eq_zero (A := A) (R := R) (reg := reg)
          (β := β) (x := x) (ν := ν) (n := n) (ω := ω) hd h_conf t ht ht0
  · rw [initial_gap_sum_eq]

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost-sure zero-dimensional version of the finite-action LinUCB regret skeleton. -/
lemma regret_ae_le_initial_gap_of_confidence_dim_eq_zero [Nonempty (Fin K)]
    (hd : d = 0)
    (h_conf : ∀ᵐ ω ∂P, LinUCBConfidenceEvent A R reg β x ν ω) :
    ∀ᵐ ω ∂P, regret ν A n ω ≤ if n = 0 then 0 else gap ν (A 0 ω) := by
  filter_upwards [h_conf] with ω h_confω
  exact regret_le_initial_gap_of_confidence_dim_eq_zero (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (ω := ω) hd h_confω

omit [IsMarkovKernel ν] in
/-- Linear realizability in zero feature dimension makes all arm gaps zero. -/
lemma gap_eq_zero_of_linear_dim_eq_zero [Nonempty (Fin K)]
    {θ : Feature d} (hd : d = 0) (h_linear : LinearMeanModel ν x θ) (a : Fin K) :
    gap ν a = 0 := by
  subst d
  rw [gap_eq_bestArm_sub]
  rw [h_linear (bestArm ν), h_linear a]
  simp [dotProduct]

omit [IsMarkovKernel ν] in
/-- Linear realizability in zero feature dimension makes cumulative regret identically zero. -/
lemma regret_eq_zero_of_linear_dim_eq_zero [Nonempty (Fin K)]
    {θ : Feature d} (hd : d = 0) (h_linear : LinearMeanModel ν x θ) :
    regret ν A n ω = 0 := by
  rw [regret_eq_sum_gap]
  simp [gap_eq_zero_of_linear_dim_eq_zero (ν := ν) (x := x) (θ := θ) hd h_linear]

omit [IsMarkovKernel ν] in
/-- A nonpositive squared feature-norm bound forces every feature vector to be zero. -/
lemma feature_eq_zero_of_featureSqNormBound_nonpos
    {L2 : ℝ} (hL2 : FeatureSqNormBound x L2) (hL2_nonpos : L2 ≤ 0) (a : Fin K) :
    x a = 0 := by
  ext i
  have hnorm_nonpos : featureSqNorm x a ≤ 0 := (hL2 a).trans hL2_nonpos
  have hcoord_sq_le_norm : (x a i) ^ 2 ≤ featureSqNorm x a := by
    rw [featureSqNorm, dotProduct]
    simpa [pow_two] using
      (Finset.single_le_sum
        (s := Finset.univ) (a := i)
        (fun j _hj ↦ mul_self_nonneg (x a j)) (Finset.mem_univ i))
  have hcoord_sq_zero : (x a i) ^ 2 = 0 :=
    le_antisymm (hcoord_sq_le_norm.trans hnorm_nonpos) (sq_nonneg (x a i))
  exact sq_eq_zero_iff.mp hcoord_sq_zero

omit [IsMarkovKernel ν] in
/-- Under linear realizability, a nonpositive feature-norm bound makes all arm gaps zero. -/
lemma gap_eq_zero_of_linear_featureSqNormBound_nonpos [Nonempty (Fin K)]
    {L2 : ℝ} {θ : Feature d}
    (hL2 : FeatureSqNormBound x L2) (hL2_nonpos : L2 ≤ 0)
    (h_linear : LinearMeanModel ν x θ) (a : Fin K) :
    gap ν a = 0 := by
  rw [gap_eq_bestArm_sub]
  rw [h_linear (bestArm ν), h_linear a]
  have hbest : x (bestArm ν) = 0 :=
    feature_eq_zero_of_featureSqNormBound_nonpos (x := x) hL2 hL2_nonpos (bestArm ν)
  have ha : x a = 0 :=
    feature_eq_zero_of_featureSqNormBound_nonpos (x := x) hL2 hL2_nonpos a
  simp [hbest, ha, dotProduct]

omit [IsMarkovKernel ν] in
/-- Under linear realizability, a nonpositive feature-norm bound makes cumulative regret zero. -/
lemma regret_eq_zero_of_linear_featureSqNormBound_nonpos [Nonempty (Fin K)]
    {L2 : ℝ} {θ : Feature d}
    (hL2 : FeatureSqNormBound x L2) (hL2_nonpos : L2 ≤ 0)
    (h_linear : LinearMeanModel ν x θ) :
    regret ν A n ω = 0 := by
  rw [regret_eq_sum_gap]
  simp [gap_eq_zero_of_linear_featureSqNormBound_nonpos (ν := ν) (x := x)
    (θ := θ) hL2 hL2_nonpos h_linear]

omit [IsMarkovKernel ν] in
/-- At horizon zero, the horizon-local self-normalized confidence event is vacuous. -/
lemma selfNormalizedConfidenceEventUpTo_of_zero_horizon
    (β : ℕ → ℝ) (hn : n = 0) :
    LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω := by
  subst n
  intro t ht _ht0 _a
  simp at ht

omit [IsMarkovKernel ν] in
/-- If a feature vector is zero, then every least-squares prediction for that arm is zero. -/
lemma estimatedReward_eq_zero_of_feature_eq_zero
    {a : Fin K} (ha : x a = 0) :
    estimatedReward A R reg x a n ω = 0 := by
  simp [estimatedReward, ha, dotProduct]

omit [IsMarkovKernel ν] in
/-- If a feature vector is zero, then its LinUCB quadratic width form is zero. -/
lemma widthQuadraticForm_eq_zero_of_feature_eq_zero
    {a : Fin K} (ha : x a = 0) :
    widthQuadraticForm A reg x a n ω = 0 := by
  simp [widthQuadraticForm, ha, dotProduct]

omit [IsMarkovKernel ν] in
/-- If a feature vector is zero, then its LinUCB width is zero. -/
lemma width_eq_zero_of_feature_eq_zero
    {a : Fin K} (ha : x a = 0) :
    width A reg x a n ω = 0 := by
  simp [width, widthQuadraticForm_eq_zero_of_feature_eq_zero (A := A) (reg := reg)
    (x := x) (n := n) (ω := ω) ha]

omit [IsMarkovKernel ν] in
/-- In zero feature dimension under linear realizability, the horizon-local self-normalized
confidence event is deterministic. -/
lemma selfNormalizedConfidenceEventUpTo_of_linear_dim_eq_zero
    (β : ℕ → ℝ) {θ : Feature d} (hd : d = 0)
    (h_linear : LinearMeanModel ν x θ) :
    LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω := by
  intro t _ht _ht0 a
  have hmean : (ν a)[id] = 0 := by
    subst d
    simpa [dotProduct] using h_linear a
  have hmean' : (∫ r, r ∂ν a) = 0 := by
    simpa using hmean
  simp [estimatedReward_eq_zero_of_dim_eq_zero (A := A) (R := R) (reg := reg)
    (x := x) (n := t) (ω := ω) hd a,
    width_eq_zero_of_dim_eq_zero (A := A) (reg := reg) (x := x) (n := t)
      (ω := ω) hd a, hmean']

omit [IsMarkovKernel ν] in
/-- Under linear realizability, a nonpositive feature-norm bound makes the horizon-local
self-normalized confidence event deterministic. -/
lemma selfNormalizedConfidenceEventUpTo_of_featureSqNormBound_nonpos
    (β : ℕ → ℝ) {L2 : ℝ} {θ : Feature d}
    (hL2 : FeatureSqNormBound x L2) (hL2_nonpos : L2 ≤ 0)
    (h_linear : LinearMeanModel ν x θ) :
    LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω := by
  intro t _ht _ht0 a
  have ha : x a = 0 :=
    feature_eq_zero_of_featureSqNormBound_nonpos (x := x) hL2 hL2_nonpos a
  have hmean : (ν a)[id] = 0 := by
    rw [h_linear a, ha]
    simp [dotProduct]
  have hmean' : (∫ r, r ∂ν a) = 0 := by
    simpa using hmean
  simp [estimatedReward_eq_zero_of_feature_eq_zero (A := A) (R := R) (reg := reg)
    (x := x) (n := t) (ω := ω) ha,
    width_eq_zero_of_feature_eq_zero (A := A) (reg := reg) (x := x) (n := t)
      (ω := ω) ha, hmean']


end LinUCB

end Bandits
