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

omit [IsMarkovKernel ν] in
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

end LinUCB

end Bandits
