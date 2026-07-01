/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: OpenAI, Fawad Haider
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.LinUCB.Regret.Degenerate

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

end LinUCB

end Bandits
