/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: OpenAI, Fawad Haider
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.LinUCB.Coordinate

/-!
# LinUCB Regret: Core Algebra

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
/-- If the LinUCB confidence inequalities hold for a comparator arm and the selected arm, and the
selected arm has maximal LinUCB index, then instantaneous regret is controlled by the selected
arm's LinUCB width. -/
lemma mean_sub_mean_arm_le_two_mul_width (a : Fin K)
    (h_best : (ν a)[id] ≤ index A R reg β x a n ω)
    (h_arm : estimatedReward A R reg x (A n ω) n ω -
        √(β (n + 1)) * width A reg x (A n ω) n ω ≤ (ν (A n ω))[id])
    (h_le : index A R reg β x a n ω ≤ index A R reg β x (A n ω) n ω) :
    (ν a)[id] - (ν (A n ω))[id] ≤
      2 * (√(β (n + 1)) * width A reg x (A n ω) n ω) := by
  rw [sub_le_iff_le_add']
  calc
    (ν a)[id] ≤ index A R reg β x a n ω := h_best
    _ ≤ index A R reg β x (A n ω) n ω := h_le
    _ ≤ (ν (A n ω))[id] +
        2 * (√(β (n + 1)) * width A reg x (A n ω) n ω) := by
      rw [index, two_mul, ← add_assoc]
      gcongr
      rwa [sub_le_iff_le_add] at h_arm

omit [IsMarkovKernel ν] in
/-- The gap of the selected arm is bounded by twice its LinUCB bonus whenever the usual confidence
inequalities hold and the selected arm has maximal LinUCB index. -/
lemma gap_arm_le_two_mul_width [Nonempty (Fin K)]
    (h_best : (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) n ω)
    (h_arm : estimatedReward A R reg x (A n ω) n ω -
        √(β (n + 1)) * width A reg x (A n ω) n ω ≤ (ν (A n ω))[id])
    (h_le : index A R reg β x (bestArm ν) n ω ≤
      index A R reg β x (A n ω) n ω) :
    gap ν (A n ω) ≤ 2 * (√(β (n + 1)) * width A reg x (A n ω) n ω) := by
  rw [gap_eq_bestArm_sub]
  exact mean_sub_mean_arm_le_two_mul_width (A := A) (R := R) (reg := reg) (β := β) (x := x)
    (ν := ν) (a := bestArm ν) h_best h_arm h_le

omit [IsMarkovKernel ν] in
lemma gap_le_two_mul_sqrt_beta_mul_sqrt_min_widthQuadraticForm
    (t : ℕ)
    (h_gap_two : gap ν (A t ω) ≤ 2)
    (h_gap_width : gap ν (A t ω) ≤
      2 * (√(β (t + 1)) * width A reg x (A t ω) t ω))
    (hβ_le : β (t + 1) ≤ β n)
    (hβn_one : 1 ≤ β n) :
    gap ν (A t ω) ≤
      2 * (√(β n) * √(min 1 (widthQuadraticForm A reg x (A t ω) t ω))) := by
  by_cases hq_le_one : widthQuadraticForm A reg x (A t ω) t ω ≤ 1
  · have hwidth_nonneg : 0 ≤ width A reg x (A t ω) t ω := Real.sqrt_nonneg _
    have hsqrt_le : √(β (t + 1)) ≤ √(β n) := Real.sqrt_le_sqrt hβ_le
    have hbonus_le :
        2 * (√(β (t + 1)) * width A reg x (A t ω) t ω) ≤
          2 * (√(β n) * width A reg x (A t ω) t ω) := by
      exact mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_right hsqrt_le hwidth_nonneg) (by norm_num)
    have hmin :
        √(min 1 (widthQuadraticForm A reg x (A t ω) t ω)) =
          width A reg x (A t ω) t ω := by
      rw [min_eq_right hq_le_one, width]
    simpa [hmin] using h_gap_width.trans hbonus_le
  · have hq_one : 1 ≤ widthQuadraticForm A reg x (A t ω) t ω := by linarith
    have hsqrt_one : 1 ≤ √(β n) := by
      simpa using (Real.one_le_sqrt).2 hβn_one
    have htwo_le :
        2 ≤ 2 * (√(β n) * √(min 1 (widthQuadraticForm A reg x (A t ω) t ω))) := by
      rw [min_eq_left hq_one, Real.sqrt_one]
      nlinarith
    exact h_gap_two.trans htwo_le

omit [IsMarkovKernel ν] in
/-- If every realized gap up to horizon `n` is bounded pointwise, then regret up to `n` is bounded
by the corresponding sum of pointwise bounds. -/
lemma regret_le_sum_of_gap_bound (B : ℕ → ℝ)
    (hB : ∀ t, t ∈ range n → gap ν (A t ω) ≤ B t) :
    regret ν A n ω ≤ ∑ t ∈ range n, B t := by
  rw [regret_eq_sum_gap]
  exact sum_le_sum hB

omit [IsMarkovKernel ν] in
/-- Pathwise cumulative-regret bound obtained by summing positive-time capped quadratic-width
terms, leaving the time-zero gap unchanged. -/
lemma regret_le_sum_sqrt_capped_width_of_forall_gap_le
    (h_gap : ∀ t, t ∈ range n → t ≠ 0 →
      gap ν (A t ω) ≤
        2 * (√(β n) * √(min 1 (widthQuadraticForm A reg x (A t ω) t ω)))) :
    regret ν A n ω ≤
      ∑ t ∈ range n,
        if t = 0 then gap ν (A 0 ω)
        else 2 * (√(β n) * √(min 1 (widthQuadraticForm A reg x (A t ω) t ω))) := by
  refine regret_le_sum_of_gap_bound (A := A) (ν := ν) (n := n) (ω := ω)
    (B := fun t ↦
      if t = 0 then gap ν (A 0 ω)
      else 2 * (√(β n) * √(min 1 (widthQuadraticForm A reg x (A t ω) t ω)))) ?_
  intro t ht
  by_cases ht0 : t = 0
  · simp [ht0]
  · simpa [ht0] using h_gap t ht ht0

lemma sum_positive_capped_bonus_le_two_mul_sqrt_nat_mul_beta_mul_sqrt_capped_sum
    (hβn_nonneg : 0 ≤ β n)
    (h_nonneg : ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm A reg x (A t ω) t ω) :
    (∑ t ∈ range n,
      if t = 0 then 0
      else 2 * (√(β n) * √(min 1 (widthQuadraticForm A reg x (A t ω) t ω)))) ≤
      2 * (√((n : ℝ) * β n) * √(cappedQuadraticWidthSum A reg x n ω)) := by
  calc
    (∑ t ∈ range n,
      if t = 0 then 0
      else 2 * (√(β n) * √(min 1 (widthQuadraticForm A reg x (A t ω) t ω))))
        = 2 * ∑ t ∈ range n,
          (if t = 0 then 0 else √(β n)) *
            (if t = 0 then 0
              else √(min 1 (widthQuadraticForm A reg x (A t ω) t ω))) := by
          rw [mul_sum]
          refine sum_congr rfl ?_
          intro t ht
          by_cases ht0 : t = 0
          · simp [ht0]
          · simp [ht0]
    _ ≤ 2 * (√(∑ t ∈ range n, (if t = 0 then 0 else √(β n)) ^ 2) *
        √(∑ t ∈ range n,
          (if t = 0 then 0
            else √(min 1 (widthQuadraticForm A reg x (A t ω) t ω))) ^ 2)) := by
      gcongr
      exact Real.sum_mul_le_sqrt_mul_sqrt (range n)
        (fun t ↦ if t = 0 then 0 else √(β n))
        (fun t ↦ if t = 0 then 0
          else √(min 1 (widthQuadraticForm A reg x (A t ω) t ω)))
    _ ≤ 2 * (√((n : ℝ) * β n) * √(cappedQuadraticWidthSum A reg x n ω)) := by
      gcongr
      · calc
          (∑ t ∈ range n, (if t = 0 then 0 else √(β n)) ^ 2)
              ≤ ∑ _t ∈ range n, β n := by
                refine sum_le_sum ?_
                intro t ht
                by_cases ht0 : t = 0
                · simp [ht0, hβn_nonneg]
                · simp [ht0, Real.sq_sqrt hβn_nonneg]
          _ = (n : ℝ) * β n := by
            simp [sum_const, nsmul_eq_mul]
      · rw [cappedQuadraticWidthSum]
        refine le_of_eq ?_
        refine sum_congr rfl ?_
        intro t ht
        by_cases ht0 : t = 0
        · simp [ht0]
        · have hmin_nonneg :
              0 ≤ min 1 (widthQuadraticForm A reg x (A t ω) t ω) := by
            exact le_min zero_le_one (h_nonneg t ht ht0)
          simp [ht0, Real.sq_sqrt hmin_nonneg]

omit [IsMarkovKernel ν] in
/-- Pathwise cumulative-regret bound using the textbook capped quadratic-width sum. -/
lemma regret_le_initial_add_sqrt_nat_mul_beta_capped_sum
    (hβn_nonneg : 0 ≤ β n)
    (h_nonneg : ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm A reg x (A t ω) t ω)
    (h_gap : ∀ t, t ∈ range n → t ≠ 0 →
      gap ν (A t ω) ≤
        2 * (√(β n) * √(min 1 (widthQuadraticForm A reg x (A t ω) t ω)))) :
    regret ν A n ω ≤
      (∑ t ∈ range n, if t = 0 then gap ν (A 0 ω) else 0) +
        2 * (√((n : ℝ) * β n) * √(cappedQuadraticWidthSum A reg x n ω)) := by
  refine (regret_le_sum_sqrt_capped_width_of_forall_gap_le (A := A) (reg := reg)
    (β := β) (x := x) (ν := ν) (n := n) (ω := ω) h_gap).trans ?_
  have hsplit :
      (∑ t ∈ range n,
        if t = 0 then gap ν (A 0 ω)
        else 2 * (√(β n) * √(min 1 (widthQuadraticForm A reg x (A t ω) t ω)))) =
        (∑ t ∈ range n, if t = 0 then gap ν (A 0 ω) else 0) +
          ∑ t ∈ range n,
            if t = 0 then 0
            else 2 * (√(β n) *
              √(min 1 (widthQuadraticForm A reg x (A t ω) t ω))) := by
    rw [← sum_add_distrib]
    refine sum_congr rfl ?_
    intro t ht
    by_cases ht0 : t = 0
    · simp [ht0]
    · simp [ht0]
  rw [hsplit]
  exact add_le_add le_rfl
    (sum_positive_capped_bonus_le_two_mul_sqrt_nat_mul_beta_mul_sqrt_capped_sum
      (A := A) (reg := reg) (β := β) (x := x) (n := n) (ω := ω)
      hβn_nonneg h_nonneg)

omit [IsMarkovKernel ν] in
/-- If the capped quadratic-width sum is bounded by `W`, the pathwise capped regret bound can use
`√W` in place of the realized capped-sum square root. -/
lemma regret_le_initial_add_sqrt_nat_mul_beta_of_capped_sum_le (W : ℝ)
    (h_regret :
      regret ν A n ω ≤
        (∑ t ∈ range n, if t = 0 then gap ν (A 0 ω) else 0) +
          2 * (√((n : ℝ) * β n) * √(cappedQuadraticWidthSum A reg x n ω)))
    (hW : cappedQuadraticWidthSum A reg x n ω ≤ W) :
    regret ν A n ω ≤
      (∑ t ∈ range n, if t = 0 then gap ν (A 0 ω) else 0) +
        2 * (√((n : ℝ) * β n) * √W) := by
  refine h_regret.trans ?_
  gcongr

/-- Minimal beta-schedule assumptions consumed by the deterministic regret proof. -/
def BetaSchedule (β : ℕ → ℝ) : Prop :=
  1 ≤ β 1 ∧ Monotone β

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Projection from `BetaSchedule`: the confidence-radius schedule starts at least at one. -/
lemma BetaSchedule.one (hβ : BetaSchedule β) : 1 ≤ β 1 :=
  hβ.1

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Projection from `BetaSchedule`: the confidence-radius schedule is monotone. -/
lemma BetaSchedule.monotone (hβ : BetaSchedule β) : Monotone β :=
  hβ.2

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- A confidence-radius schedule with `1 ≤ β 1` and monotone `β` is nonnegative at every positive
horizon. -/
lemma beta_nonneg_of_one_le_of_monotone
    (hβ_one : 1 ≤ β 1) (hβ_mono : Monotone β) {n : ℕ} (hn : n ≠ 0) :
    0 ≤ β n := by
  have hn_one : 1 ≤ n := Nat.succ_le_iff.mpr (Nat.pos_of_ne_zero hn)
  exact ((zero_le_one : (0 : ℝ) ≤ 1).trans hβ_one).trans (hβ_mono hn_one)

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- A `BetaSchedule` is nonnegative at every positive horizon. -/
lemma BetaSchedule.nonneg_of_ne_zero (hβ : BetaSchedule β) {n : ℕ} (hn : n ≠ 0) :
    0 ≤ β n :=
  beta_nonneg_of_one_le_of_monotone (β := β) hβ.one hβ.monotone hn

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- The finite-horizon textbook beta envelope satisfies the schedule assumptions consumed by the
regret proof. -/
lemma textbookLinUCBBeta_schedule
    (d : ℕ) (reg S2 : ℝ) (σ2 : ℝ≥0) (L2 : ℝ) (n : ℕ) (δ : ℝ) :
    BetaSchedule (textbookLinUCBBeta d reg S2 σ2 L2 n δ) := by
  let b : ℝ := textbookLinUCBConfidenceRadius d reg S2 σ2 L2 n δ
  have hb : 1 ≤ b := by
    exact le_max_left 1 _
  constructor
  · have hoff_nonneg : 0 ≤ max 0 ((1 : ℝ) - (n : ℝ)) := le_max_left _ _
    have hb_ext : 1 ≤ b + max 0 ((1 : ℝ) - (n : ℝ)) :=
      hb.trans (le_add_of_nonneg_right hoff_nonneg)
    simpa [textbookLinUCBBeta, b] using hb_ext
  · intro m m' hmm'
    have hcast : (m : ℝ) ≤ (m' : ℝ) := by exact_mod_cast hmm'
    have hoff_le :
        max 0 ((m : ℝ) - (n : ℝ)) ≤ max 0 ((m' : ℝ) - (n : ℝ)) :=
      max_le_max_left 0 (sub_le_sub_right hcast _)
    simpa [textbookLinUCBBeta, b] using add_le_add_left hoff_le b

omit [IsMarkovKernel ν] in
lemma initial_gap_sum_eq :
    (∑ t ∈ range n, if t = 0 then gap ν (A 0 ω) else 0) =
      if n = 0 then 0 else gap ν (A 0 ω) := by
  cases n <;> simp


end LinUCB

end Bandits
