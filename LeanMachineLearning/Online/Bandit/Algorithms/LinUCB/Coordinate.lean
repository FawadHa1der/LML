/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: OpenAI, Fawad Haider
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.LinUCB.Concentration

/-!
# LinUCB for finite-action linear bandits
Chapter 19 of *Bandit Algorithms*:
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

section AlgorithmBehavior

omit [IsMarkovKernel ν] in
/-- Failure-probability transfer from the textbook determinant-ratio self-normalized event to the
centered-noise-plus-bias event. -/
lemma probReal_centeredNoiseBiasUpTo_failure_le_of_textbookNoiseUpTo_failure_le
    (θ : Feature d) (S2 : ℝ) {σ2 : ℝ≥0} {δ : ℝ}
    (hreg_pos : 0 < reg)
    (hθ : ParameterSqNormBound θ S2)
    (h_budget : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      (√(textbookSelfNormalizedNoiseBound σ2 δ (designDetRatio A reg x t ω)) +
        √(reg * S2)) ^ 2 ≤ β (t + 1))
    (h_noise_failure :
      P.real {ω |
        ¬ LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} ≤ δ) :
    P.real {ω | ¬ LinUCBCenteredNoiseBiasConfidenceEventUpTo A R reg β x ν θ n ω} ≤ δ := by
  refine le_trans ?_ h_noise_failure
  refine probReal_failure_le_of_ae_imp (P := P) ?_
  filter_upwards [h_budget] with ω h_budgetω h_noiseω
  exact LinUCBCenteredNoiseBiasConfidenceEventUpTo.of_textbookSelfNormalizedNoise
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
    (ω := ω) θ S2 hreg_pos hθ h_noiseω h_budgetω

omit [IsMarkovKernel ν] in
/-- Probability monotonicity for the coordinate-wise route to the centered-noise event. -/
lemma probReal_centeredNoiseCoordinateBoundEventUpTo_le_centeredNoiseConfidenceEventUpTo
    (hreg_pos : 0 < reg)
    {coordBudget noiseBudget : ℕ → ℝ}
    (hcoord_nonneg : ∀ t, t ∈ range n → t ≠ 0 → 0 ≤ coordBudget (t + 1))
    (h_budget : ∀ t, t ∈ range n → t ≠ 0 →
      (d : ℝ) * coordBudget (t + 1) ^ 2 / reg ≤ noiseBudget (t + 1)) :
    P.real {ω | LinUCBCenteredNoiseCoordinateBoundEventUpTo A R coordBudget x ν n ω} ≤
      P.real {ω | LinUCBCenteredNoiseConfidenceEventUpTo A R reg noiseBudget x ν n ω} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω hcoordω ↦
    LinUCBCenteredNoiseConfidenceEventUpTo.of_coordinateBound (A := A) (R := R)
      (reg := reg) (x := x) (ν := ν) (n := n) (ω := ω) hreg_pos hcoordω
      hcoord_nonneg h_budget

omit [IsMarkovKernel ν] in
/-- High-probability transfer from the coordinate-wise centered-noise event to the centered-noise
quadratic-form event. -/
lemma probReal_centeredNoiseConfidenceEventUpTo_ge_of_coordinateBoundEventUpTo_ge
    (hreg_pos : 0 < reg)
    {coordBudget noiseBudget : ℕ → ℝ} {δ : ℝ}
    (hcoord_nonneg : ∀ t, t ∈ range n → t ≠ 0 → 0 ≤ coordBudget (t + 1))
    (h_budget : ∀ t, t ∈ range n → t ≠ 0 →
      (d : ℝ) * coordBudget (t + 1) ^ 2 / reg ≤ noiseBudget (t + 1))
    (hcoord_prob :
      1 - δ ≤ P.real {ω |
        LinUCBCenteredNoiseCoordinateBoundEventUpTo A R coordBudget x ν n ω}) :
    1 - δ ≤
      P.real {ω | LinUCBCenteredNoiseConfidenceEventUpTo A R reg noiseBudget x ν n ω} :=
  hcoord_prob.trans
    (probReal_centeredNoiseCoordinateBoundEventUpTo_le_centeredNoiseConfidenceEventUpTo
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) (P := P)
      hreg_pos hcoord_nonneg h_budget)

omit [IsMarkovKernel ν] in
/-- Failure-probability transfer from the coordinate-wise centered-noise event to the
centered-noise quadratic-form event. -/
lemma probReal_centeredNoiseConfidenceEventUpTo_failure_le_of_coordinateBoundEventUpTo_failure_le
    (hreg_pos : 0 < reg)
    {coordBudget noiseBudget : ℕ → ℝ} {δ : ℝ}
    (hcoord_nonneg : ∀ t, t ∈ range n → t ≠ 0 → 0 ≤ coordBudget (t + 1))
    (h_budget : ∀ t, t ∈ range n → t ≠ 0 →
      (d : ℝ) * coordBudget (t + 1) ^ 2 / reg ≤ noiseBudget (t + 1))
    (hcoord_failure :
      P.real {ω | ¬ LinUCBCenteredNoiseCoordinateBoundEventUpTo A R coordBudget x ν n ω} ≤ δ) :
    P.real {ω | ¬ LinUCBCenteredNoiseConfidenceEventUpTo A R reg noiseBudget x ν n ω} ≤ δ := by
  refine le_trans ?_ hcoord_failure
  refine probReal_failure_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω hcoordω ↦
    LinUCBCenteredNoiseConfidenceEventUpTo.of_coordinateBound (A := A) (R := R)
      (reg := reg) (x := x) (ν := ν) (n := n) (ω := ω) hreg_pos hcoordω
      hcoord_nonneg h_budget

omit [IsMarkovKernel ν] in
/-- Failure probability of the coordinate-wise centered-noise event is bounded by the sum of all
one-coordinate upper- and lower-tail failures through the horizon. -/
lemma probReal_centeredNoiseCoordinateBoundEventUpTo_failure_le_sum
    {coordBudget : ℕ → ℝ} :
    P.real {ω | ¬ LinUCBCenteredNoiseCoordinateBoundEventUpTo A R coordBudget x ν n ω} ≤
      ∑ t ∈ range n, if t = 0 then 0 else
        ∑ i : Fin d,
          (P.real (centeredNoiseCoordinateUpperFailure A R coordBudget x ν t i) +
            P.real (centeredNoiseCoordinateLowerFailure A R coordBudget x ν t i)) := by
  classical
  let badAt : ℕ → Set Ω := fun t ↦
    if t = 0 then ∅ else
      ⋃ i : Fin d,
        centeredNoiseCoordinateUpperFailure A R coordBudget x ν t i ∪
          centeredNoiseCoordinateLowerFailure A R coordBudget x ν t i
  have h_failure_subset :
      P.real {ω | ¬ LinUCBCenteredNoiseCoordinateBoundEventUpTo A R coordBudget x ν n ω} ≤
        P.real (⋃ t ∈ range n, badAt t) := by
    change P.real {ω | ¬ LinUCBCenteredNoiseCoordinateBoundEventUpTo A R coordBudget x ν n ω} ≤
      P.real {ω | ω ∈ ⋃ t ∈ range n, badAt t}
    refine probReal_event_le_of_ae_imp (P := P) ?_
    exact Filter.Eventually.of_forall fun ω h_failure ↦ by
      rw [LinUCBCenteredNoiseCoordinateBoundEventUpTo] at h_failure
      push Not at h_failure
      rcases h_failure with ⟨t, ht, ht0, i, hcoord_failure⟩
      refine Set.mem_iUnion.2 ⟨t, ?_⟩
      refine Set.mem_iUnion.2 ⟨ht, ?_⟩
      dsimp [badAt]
      rw [if_neg ht0]
      refine Set.mem_iUnion.2 ⟨i, ?_⟩
      have h_abs_failure :
          coordBudget (t + 1) < |centeredResponseVector A R ν x t ω i| :=
        hcoord_failure
      rcases (lt_abs.mp h_abs_failure) with h_upper | h_lower
      · exact Or.inl (by
          dsimp [centeredNoiseCoordinateUpperFailure]
          exact h_upper)
      · exact Or.inr (by
          dsimp [centeredNoiseCoordinateLowerFailure]
          linarith)
  calc
    P.real {ω | ¬ LinUCBCenteredNoiseCoordinateBoundEventUpTo A R coordBudget x ν n ω}
        ≤ P.real (⋃ t ∈ range n, badAt t) := h_failure_subset
    _ ≤ ∑ t ∈ range n, P.real (badAt t) :=
        probReal_biUnion_finset_le_sum (P := P) (range n) badAt
    _ ≤ ∑ t ∈ range n, if t = 0 then 0 else
        ∑ i : Fin d,
          (P.real (centeredNoiseCoordinateUpperFailure A R coordBudget x ν t i) +
            P.real (centeredNoiseCoordinateLowerFailure A R coordBudget x ν t i)) := by
        refine Finset.sum_le_sum fun t ht ↦ ?_
        by_cases ht0 : t = 0
        · simp [badAt, ht0, measureReal_def]
        · dsimp [badAt]
          rw [if_neg ht0]
          calc
            P.real (⋃ i : Fin d,
                centeredNoiseCoordinateUpperFailure A R coordBudget x ν t i ∪
                  centeredNoiseCoordinateLowerFailure A R coordBudget x ν t i)
                ≤ ∑ i : Fin d,
                    P.real (centeredNoiseCoordinateUpperFailure A R coordBudget x ν t i ∪
                      centeredNoiseCoordinateLowerFailure A R coordBudget x ν t i) :=
              probReal_iUnion_fintype_le_sum (P := P) fun i : Fin d ↦
                centeredNoiseCoordinateUpperFailure A R coordBudget x ν t i ∪
                  centeredNoiseCoordinateLowerFailure A R coordBudget x ν t i
            _ ≤ ∑ i : Fin d,
                (P.real (centeredNoiseCoordinateUpperFailure A R coordBudget x ν t i) +
                  P.real (centeredNoiseCoordinateLowerFailure A R coordBudget x ν t i)) := by
              simpa using
                (Finset.sum_le_sum (s := Finset.univ) fun i _hi ↦
                  probReal_union_le (P := P)
                    (centeredNoiseCoordinateUpperFailure A R coordBudget x ν t i)
                    (centeredNoiseCoordinateLowerFailure A R coordBudget x ν t i))
            _ = if t = 0 then 0 else
                ∑ i : Fin d,
                  (P.real (centeredNoiseCoordinateUpperFailure A R coordBudget x ν t i) +
                    P.real (centeredNoiseCoordinateLowerFailure A R coordBudget x ν t i)) := by
              simp [ht0]

omit [IsMarkovKernel ν] in
/-- Scalar coordinate tail bound used by the conservative coordinate-wise concentration route. -/
noncomputable def centeredNoiseCoordinateTailBound
    (σ2 : ℝ≥0) (L2 : ℝ) (t : ℕ) (B : ℝ) : ℝ :=
  Real.exp
    (-B ^ 2 /
      (2 * (∑ _s ∈ range t, (⟨(√L2) ^ 2, sq_nonneg (√L2)⟩ * σ2 : ℝ≥0))))

omit [IsMarkovKernel ν] in
/-- Variance proxy in the scalar coordinate tail bound. -/
noncomputable def centeredNoiseCoordinateVarianceSum
    (σ2 : ℝ≥0) (L2 : ℝ) (t : ℕ) : ℝ :=
  ∑ _s ∈ range t, ((⟨(√L2) ^ 2, sq_nonneg (√L2)⟩ * σ2 : ℝ≥0) : ℝ)

omit [IsMarkovKernel ν] in
/-- Concrete coordinate log-budget that makes one scalar coordinate tail at most
`δ / (2 n d)`.

The index is a beta-style time index: at process time `t`, callers use budget index `t + 1`, so
the variance sum is taken over `range t`. -/
noncomputable def centeredNoiseCoordinateLogBudget
    (σ2 : ℝ≥0) (L2 : ℝ) (n d : ℕ) (δ : ℝ) (m : ℕ) : ℝ :=
  √(2 * centeredNoiseCoordinateVarianceSum σ2 L2 (m - 1) *
    Real.log (max 1 (2 * (n : ℝ) * (d : ℝ) / δ)))

omit [IsMarkovKernel ν] in
/-- The coordinate log-budget is nonnegative. -/
lemma centeredNoiseCoordinateLogBudget_nonneg
    (σ2 : ℝ≥0) (L2 : ℝ) (n d : ℕ) (δ : ℝ) (m : ℕ) :
    0 ≤ centeredNoiseCoordinateLogBudget σ2 L2 n d δ m :=
  Real.sqrt_nonneg _

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- The variance proxy is positive at positive times when the feature bound and subgaussian proxy
are positive. -/
lemma centeredNoiseCoordinateVarianceSum_pos
    {σ2 : ℝ≥0} {L2 : ℝ} {t : ℕ}
    (hσ2_pos : 0 < (σ2 : ℝ)) (hL2_pos : 0 < L2) (ht : t ≠ 0) :
    0 < centeredNoiseCoordinateVarianceSum σ2 L2 t := by
  have hterm_pos :
      0 < ((⟨(√L2) ^ 2, sq_nonneg (√L2)⟩ * σ2 : ℝ≥0) : ℝ) := by
    rw [NNReal.coe_mul]
    exact mul_pos (sq_pos_of_pos (Real.sqrt_pos.2 hL2_pos)) hσ2_pos
  rw [centeredNoiseCoordinateVarianceSum]
  exact Finset.sum_pos (fun _ _ ↦ hterm_pos) (Finset.nonempty_range_iff.mpr ht)

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- The concrete coordinate log-budget discharges the per-coordinate tail condition used by the
coordinate-union concentration route. -/
lemma centeredNoiseCoordinateTailBound_logBudget_le
    {σ2 : ℝ≥0} {L2 δ : ℝ} {n d t : ℕ}
    (hσ2_pos : 0 < (σ2 : ℝ)) (hL2_pos : 0 < L2)
    (hδ_pos : 0 < δ) (hn : n ≠ 0) (hd : d ≠ 0) (ht : t ≠ 0) :
    centeredNoiseCoordinateTailBound σ2 L2 t
        (centeredNoiseCoordinateLogBudget σ2 L2 n d δ (t + 1)) ≤
      δ / (2 * (n : ℝ) * (d : ℝ)) := by
  let V := centeredNoiseCoordinateVarianceSum σ2 L2 t
  let M := max 1 (2 * (n : ℝ) * (d : ℝ) / δ)
  have hV_pos : 0 < V := by
    simpa [V] using centeredNoiseCoordinateVarianceSum_pos (σ2 := σ2) (L2 := L2)
      hσ2_pos hL2_pos ht
  have hM_ge_one : 1 ≤ M := by
    simp [M]
  have hM_pos : 0 < M := lt_of_lt_of_le zero_lt_one hM_ge_one
  have hlog_nonneg : 0 ≤ Real.log M := Real.log_nonneg hM_ge_one
  have hbudget_sq :
      centeredNoiseCoordinateLogBudget σ2 L2 n d δ (t + 1) ^ 2 =
        2 * V * Real.log M := by
    rw [centeredNoiseCoordinateLogBudget, Real.sq_sqrt]
    · simp [V, M]
    · exact mul_nonneg (mul_nonneg (by positivity) hV_pos.le) hlog_nonneg
  have htail_eq :
      centeredNoiseCoordinateTailBound σ2 L2 t
          (centeredNoiseCoordinateLogBudget σ2 L2 n d δ (t + 1)) =
        M⁻¹ := by
    rw [centeredNoiseCoordinateTailBound, hbudget_sq]
    have hden :
        (((∑ _s ∈ range t,
            (⟨(√L2) ^ 2, sq_nonneg (√L2)⟩ * σ2 : ℝ≥0)) : ℝ)) = V := by
      simp [V, centeredNoiseCoordinateVarianceSum]
    rw [NNReal.coe_sum]
    rw [hden]
    change Real.exp (-(2 * V * Real.log M) / (2 * V)) = M⁻¹
    have harg : -(2 * V * Real.log M) / (2 * V) = -Real.log M := by
      field_simp [hV_pos.ne']
    rw [harg, Real.exp_neg, Real.exp_log hM_pos]
  rw [htail_eq]
  have hratio_pos : 0 < 2 * (n : ℝ) * (d : ℝ) / δ := by
    positivity
  have hratio_le_M : 2 * (n : ℝ) * (d : ℝ) / δ ≤ M := by
    simp [M]
  have hinv_le : M⁻¹ ≤ (2 * (n : ℝ) * (d : ℝ) / δ)⁻¹ := by
    simpa [one_div] using one_div_le_one_div_of_le hratio_pos hratio_le_M
  refine hinv_le.trans_eq ?_
  field_simp [hδ_pos.ne', Nat.cast_ne_zero.mpr hn, Nat.cast_ne_zero.mpr hd]

omit [IsMarkovKernel ν] in
/-- Total finite-union failure budget for the conservative coordinate-wise centered-noise event. -/
noncomputable def centeredNoiseCoordinateTailFailureSum
    (d : ℕ) (σ2 : ℝ≥0) (L2 : ℝ) (coordBudget : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ t ∈ range n, if t = 0 then 0 else
    ∑ _i : Fin d, 2 * centeredNoiseCoordinateTailBound σ2 L2 t (coordBudget (t + 1))

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- If every positive time receives at most a `δ / n` share of coordinate-tail failure
probability, then the total finite-union coordinate-tail failure budget is at most `δ`.

This is the finite-union accounting step in the conservative coordinate route. It separates the
probabilistic scalar tail bound at each time from the deterministic bookkeeping over the horizon. -/
lemma centeredNoiseCoordinateTailFailureSum_le_of_time_tail_le
    {σ2 : ℝ≥0} {L2 : ℝ} {coordBudget : ℕ → ℝ} {δ : ℝ}
    (hδ_nonneg : 0 ≤ δ)
    (h_tail : ∀ t, t ∈ range n → t ≠ 0 →
      (∑ _i : Fin d, 2 * centeredNoiseCoordinateTailBound σ2 L2 t
        (coordBudget (t + 1))) ≤ δ / (n : ℝ)) :
    centeredNoiseCoordinateTailFailureSum d σ2 L2 coordBudget n ≤ δ := by
  by_cases hn : n = 0
  · simp [centeredNoiseCoordinateTailFailureSum, hn, hδ_nonneg]
  · have hn_pos : 0 < (n : ℝ) := by
      exact_mod_cast Nat.pos_of_ne_zero hn
    rw [centeredNoiseCoordinateTailFailureSum]
    calc
      (∑ t ∈ range n, if t = 0 then 0 else
          ∑ _i : Fin d, 2 * centeredNoiseCoordinateTailBound σ2 L2 t
            (coordBudget (t + 1)))
          ≤ ∑ _t ∈ range n, δ / (n : ℝ) := by
            refine Finset.sum_le_sum fun t ht ↦ ?_
            by_cases ht0 : t = 0
            · rw [if_pos ht0]
              exact div_nonneg hδ_nonneg hn_pos.le
            · rw [if_neg ht0]
              exact h_tail t ht ht0
      _ = (n : ℝ) * (δ / (n : ℝ)) := by
            simp [sum_const, nsmul_eq_mul]
      _ = δ := by
            field_simp [ne_of_gt hn_pos]

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- If each coordinate tail at each positive time is at most `δ / (2 n d)`, then the total
finite-union coordinate-tail failure budget is at most `δ`.

The assumptions `n ≠ 0` and `d ≠ 0` are only for this concrete per-coordinate budget. The
zero-horizon and zero-dimensional cases are handled separately by the deterministic regret
lemmas. -/
lemma centeredNoiseCoordinateTailFailureSum_le_of_coord_tail_le
    {σ2 : ℝ≥0} {L2 : ℝ} {coordBudget : ℕ → ℝ} {δ : ℝ}
    (hδ_nonneg : 0 ≤ δ) (hn : n ≠ 0) (hd : d ≠ 0)
    (h_tail : ∀ t, t ∈ range n → t ≠ 0 →
      centeredNoiseCoordinateTailBound σ2 L2 t (coordBudget (t + 1)) ≤
        δ / (2 * (n : ℝ) * (d : ℝ))) :
    centeredNoiseCoordinateTailFailureSum d σ2 L2 coordBudget n ≤ δ := by
  refine centeredNoiseCoordinateTailFailureSum_le_of_time_tail_le
    (d := d) (n := n) (σ2 := σ2) (L2 := L2) (coordBudget := coordBudget)
    hδ_nonneg ?_
  intro t ht ht0
  calc
    (∑ _i : Fin d, 2 * centeredNoiseCoordinateTailBound σ2 L2 t
        (coordBudget (t + 1)))
        ≤ ∑ _i : Fin d, 2 * (δ / (2 * (n : ℝ) * (d : ℝ))) := by
          refine Finset.sum_le_sum fun i _hi ↦ ?_
          exact mul_le_mul_of_nonneg_left (h_tail t ht ht0) (by positivity)
    _ = (d : ℝ) * (2 * (δ / (2 * (n : ℝ) * (d : ℝ)))) := by
          simp [sum_const, nsmul_eq_mul]
    _ = δ / (n : ℝ) := by
          field_simp [Nat.cast_ne_zero.mpr hn, Nat.cast_ne_zero.mpr hd]

omit [IsMarkovKernel ν] in
/-- Centered-noise quadratic-form budget induced by a coordinate-wise centered-noise budget. -/
noncomputable def centeredNoiseCoordinateNoiseBudget
    (d : ℕ) (reg : ℝ) (coordBudget : ℕ → ℝ) (t : ℕ) : ℝ :=
  (d : ℝ) * coordBudget t ^ 2 / reg

omit [IsMarkovKernel ν] in
/-- LinUCB beta radius required after adding ridge bias to a coordinate-derived noise budget. -/
noncomputable def centeredNoiseCoordinateBetaBudget
    (d : ℕ) (reg S2 : ℝ) (coordBudget : ℕ → ℝ) (t : ℕ) : ℝ :=
  (√(centeredNoiseCoordinateNoiseBudget d reg coordBudget t) + √(reg * S2)) ^ 2

omit [IsMarkovKernel ν] in
/-- Conservative coordinate-union LinUCB beta schedule for a fixed horizon and failure budget.

This is not the textbook determinant-ratio beta. It packages the scalar-coordinate route: use the
explicit coordinate log-budget to control each coordinate of the centered noise vector, convert that
coordinate event into a quadratic-form noise budget, and add the ridge-bias radius. -/
noncomputable def coordinateLinUCBBeta
    (d : ℕ) (reg S2 : ℝ) (σ2 : ℝ≥0) (L2 : ℝ) (n : ℕ) (δ : ℝ) : ℕ → ℝ :=
  centeredNoiseCoordinateBetaBudget d reg S2
    (centeredNoiseCoordinateLogBudget σ2 L2 n d δ)

omit [IsMarkovKernel ν] in
/-- Monotone conservative coordinate-union LinUCB beta schedule.

The raw coordinate radius need not be convenient to prove monotone. This cumulative envelope starts
at least at one, is monotone by construction, and dominates `coordinateLinUCBBeta` at every time.
It is a conservative schedule for the scalar-coordinate route, not the textbook determinant-ratio
schedule. -/
noncomputable def coordinateLinUCBMonotoneBeta
    (d : ℕ) (reg S2 : ℝ) (σ2 : ℝ≥0) (L2 : ℝ) (n : ℕ) (δ : ℝ) : ℕ → ℝ :=
  fun m ↦
    1 + ∑ k ∈ range (m + 1),
      max 0 (coordinateLinUCBBeta d reg S2 σ2 L2 n δ k)

/-- Upper-tail probability bound for one coordinate of the centered-noise vector. -/
lemma probReal_centeredNoiseCoordinateUpperFailure_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    {coordBudget : ℕ → ℝ} (t : ℕ) (i : Fin d)
    (hcoord_nonneg : 0 ≤ coordBudget (t + 1)) :
    P.real (centeredNoiseCoordinateUpperFailure A R coordBudget x ν t i) ≤
      centeredNoiseCoordinateTailBound σ2 L2 t (coordBudget (t + 1)) := by
  have h_subset :
      P.real (centeredNoiseCoordinateUpperFailure A R coordBudget x ν t i) ≤
        P.real {ω |
          coordBudget (t + 1) ≤
            dotProduct (coordinateDirection i) (centeredResponseVector A R ν x t ω)} := by
    change P.real {ω | coordBudget (t + 1) < centeredResponseVector A R ν x t ω i} ≤
      P.real {ω |
        coordBudget (t + 1) ≤
          dotProduct (coordinateDirection i) (centeredResponseVector A R ν x t ω)}
    refine probReal_event_le_of_ae_imp (P := P) ?_
    exact Filter.Eventually.of_forall fun ω hω ↦ by
      rw [dotProduct_coordinateDirection]
      exact le_of_lt hω
  refine h_subset.trans ?_
  simpa [centeredNoiseCoordinateTailBound] using
    (probReal_dotProduct_centeredResponseVector_ge_le_of_abs_le (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) h hν (coordinateDirection i) (√L2)
      (Real.sqrt_nonneg L2)
      (abs_dotProduct_coordinateDirection_feature_le_sqrt (x := x) hL2 i) t
      hcoord_nonneg)

/-- Lower-tail probability bound for one coordinate of the centered-noise vector. -/
lemma probReal_centeredNoiseCoordinateLowerFailure_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    {coordBudget : ℕ → ℝ} (t : ℕ) (i : Fin d)
    (hcoord_nonneg : 0 ≤ coordBudget (t + 1)) :
    P.real (centeredNoiseCoordinateLowerFailure A R coordBudget x ν t i) ≤
      centeredNoiseCoordinateTailBound σ2 L2 t (coordBudget (t + 1)) := by
  have h_subset :
      P.real (centeredNoiseCoordinateLowerFailure A R coordBudget x ν t i) ≤
        P.real {ω |
          coordBudget (t + 1) ≤
            dotProduct (-coordinateDirection i) (centeredResponseVector A R ν x t ω)} := by
    change P.real {ω | centeredResponseVector A R ν x t ω i < -coordBudget (t + 1)} ≤
      P.real {ω |
        coordBudget (t + 1) ≤
          dotProduct (-coordinateDirection i) (centeredResponseVector A R ν x t ω)}
    refine probReal_event_le_of_ae_imp (P := P) ?_
    exact Filter.Eventually.of_forall fun ω hω ↦ by
      rw [dotProduct_neg_coordinateDirection]
      linarith
  refine h_subset.trans ?_
  have hQ_bound : ∀ a, |dotProduct (-coordinateDirection i) (x a)| ≤ √L2 := by
    intro a
    have hbase := abs_dotProduct_coordinateDirection_feature_le_sqrt (x := x) hL2 i a
    simpa [neg_dotProduct] using hbase
  simpa [centeredNoiseCoordinateTailBound] using
    (probReal_dotProduct_centeredResponseVector_ge_le_of_abs_le (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) h hν (-coordinateDirection i) (√L2)
      (Real.sqrt_nonneg L2) hQ_bound t hcoord_nonneg)

/-- Coordinate-wise centered-noise event failure bound derived from scalar coordinate
concentration and a finite union bound over time and coordinates. -/
lemma probReal_centeredNoiseCoordinateBoundEventUpTo_failure_le_tail_sum
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    {coordBudget : ℕ → ℝ}
    (hcoord_nonneg : ∀ t, t ∈ range n → t ≠ 0 → 0 ≤ coordBudget (t + 1)) :
    P.real {ω | ¬ LinUCBCenteredNoiseCoordinateBoundEventUpTo A R coordBudget x ν n ω} ≤
      centeredNoiseCoordinateTailFailureSum d σ2 L2 coordBudget n := by
  rw [centeredNoiseCoordinateTailFailureSum]
  refine (probReal_centeredNoiseCoordinateBoundEventUpTo_failure_le_sum
    (A := A) (R := R) (x := x) (ν := ν) (n := n) (P := P)).trans ?_
  refine Finset.sum_le_sum fun t ht ↦ ?_
  by_cases ht0 : t = 0
  · simp [ht0]
  · simp only [ht0, if_false]
    refine Finset.sum_le_sum fun i _hi ↦ ?_
    calc
      P.real (centeredNoiseCoordinateUpperFailure A R coordBudget x ν t i) +
          P.real (centeredNoiseCoordinateLowerFailure A R coordBudget x ν t i)
          ≤ centeredNoiseCoordinateTailBound σ2 L2 t (coordBudget (t + 1)) +
              centeredNoiseCoordinateTailBound σ2 L2 t (coordBudget (t + 1)) := by
            exact add_le_add
              (probReal_centeredNoiseCoordinateUpperFailure_le (A := A) (R := R)
                (reg := reg) (β := β) (x := x) (ν := ν) h hν L2 hL2 t i
                (hcoord_nonneg t ht ht0))
              (probReal_centeredNoiseCoordinateLowerFailure_le (A := A) (R := R)
                (reg := reg) (β := β) (x := x) (ν := ν) h hν L2 hL2 t i
                (hcoord_nonneg t ht ht0))
      _ = 2 * centeredNoiseCoordinateTailBound σ2 L2 t (coordBudget (t + 1)) := by ring

omit [IsMarkovKernel ν] in
/-- The horizon-local centered-noise-plus-bias event is contained in the horizon-local parameter
ellipsoid event. -/
lemma probReal_centeredNoiseBiasConfidenceEventUpTo_le_parameterEllipsoidConfidenceEventUpTo
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg) :
    P.real {ω | LinUCBCenteredNoiseBiasConfidenceEventUpTo A R reg β x ν θ n ω} ≤
      P.real {ω | LinUCBParameterEllipsoidConfidenceEventUpTo A R reg β x θ n ω} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_noiseω ↦
    LinUCBParameterEllipsoidConfidenceEventUpTo.of_centeredNoiseBias (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (ω := ω)
      θ h_linear hreg_pos h_noiseω

omit [IsMarkovKernel ν] in
/-- The horizon-local parameter ellipsoid event is contained in the horizon-local self-normalized
prediction-error event. -/
lemma probReal_parameterEllipsoidConfidenceEventUpTo_le_selfNormalizedConfidenceEventUpTo
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg) :
    P.real {ω | LinUCBParameterEllipsoidConfidenceEventUpTo A R reg β x θ n ω} ≤
      P.real {ω | LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_ellipsoidω ↦
    LinUCBSelfNormalizedConfidenceEventUpTo.of_parameterEllipsoid (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (ω := ω)
      θ h_linear hreg_pos h_ellipsoidω

omit [IsMarkovKernel ν] in
/-- High-probability bridge from the horizon-local parameter ellipsoid event to the horizon-local
self-normalized prediction-error event. -/
lemma probReal_selfNormalizedConfidenceEventUpTo_ge_of_parameterEllipsoidConfidenceEventUpTo_ge
    (θ : Feature d) {δ : ℝ}
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg)
    (h_ellipsoid_prob :
      1 - δ ≤ P.real {ω | LinUCBParameterEllipsoidConfidenceEventUpTo A R reg β x θ n ω}) :
    1 - δ ≤ P.real {ω | LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω} :=
  h_ellipsoid_prob.trans
    (probReal_parameterEllipsoidConfidenceEventUpTo_le_selfNormalizedConfidenceEventUpTo
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      (P := P) θ h_linear hreg_pos)

omit [IsMarkovKernel ν] in
/-- High-probability bridge from the horizon-local centered-noise-plus-bias event to the
horizon-local self-normalized prediction-error event. -/
lemma probReal_selfNormalizedConfidenceEventUpTo_ge_of_centeredNoiseBiasConfidenceEventUpTo_ge
    (θ : Feature d) {δ : ℝ}
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg)
    (h_noise_prob :
      1 - δ ≤
        P.real {ω | LinUCBCenteredNoiseBiasConfidenceEventUpTo A R reg β x ν θ n ω}) :
    1 - δ ≤ P.real {ω | LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω} :=
  h_noise_prob.trans
    ((probReal_centeredNoiseBiasConfidenceEventUpTo_le_parameterEllipsoidConfidenceEventUpTo
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      (P := P) θ h_linear hreg_pos).trans
      (probReal_parameterEllipsoidConfidenceEventUpTo_le_selfNormalizedConfidenceEventUpTo
        (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
        (P := P) θ h_linear hreg_pos))

omit [IsMarkovKernel ν] in
/-- Failure-probability bridge from the horizon-local parameter ellipsoid event to the
horizon-local self-normalized prediction-error event. -/
lemma probReal_selfNormalizedConfidenceEventUpTo_failure_le_of_parameterEllipsoid_failure_le
    (θ : Feature d) {δ : ℝ}
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg)
    (h_ellipsoid_failure :
      P.real {ω | ¬ LinUCBParameterEllipsoidConfidenceEventUpTo A R reg β x θ n ω} ≤ δ) :
    P.real {ω | ¬ LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω} ≤ δ := by
  refine le_trans ?_ h_ellipsoid_failure
  refine probReal_failure_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_ellipsoidω ↦
    LinUCBSelfNormalizedConfidenceEventUpTo.of_parameterEllipsoid (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (ω := ω)
      θ h_linear hreg_pos h_ellipsoidω

omit [IsMarkovKernel ν] in
/-- Failure-probability bridge from the horizon-local centered-noise-plus-bias event to the
horizon-local self-normalized prediction-error event. -/
lemma probReal_selfNormalizedConfidenceEventUpTo_failure_le_of_centeredNoiseBias_failure_le
    (θ : Feature d) {δ : ℝ}
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg)
    (h_noise_failure :
      P.real {ω | ¬ LinUCBCenteredNoiseBiasConfidenceEventUpTo A R reg β x ν θ n ω} ≤ δ) :
    P.real {ω | ¬ LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω} ≤ δ := by
  refine le_trans ?_ h_noise_failure
  refine probReal_failure_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_noiseω ↦
    LinUCBSelfNormalizedConfidenceEventUpTo.of_centeredNoiseBias (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (ω := ω)
      θ h_linear hreg_pos h_noiseω

omit [IsMarkovKernel ν] in
/-- Uniform bound on arm gaps, used as the finite-action analogue of the textbook bounded
instantaneous-regret assumption. -/
def GapBound (ν : Kernel (Fin K) ℝ) (G : ℝ) : Prop :=
  ∀ a, gap ν a ≤ G

omit [IsMarkovKernel ν] in
/-- Uniform bound on arm means. For finite-action linear bandits this is a convenient way to state
the usual bounded expected-reward assumption, for example `(ν a)[id] ∈ [-1, 1]`. -/
def MeanRewardBound (ν : Kernel (Fin K) ℝ) (lo hi : ℝ) : Prop :=
  ∀ a, lo ≤ (ν a)[id] ∧ (ν a)[id] ≤ hi

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Linear realizability plus normalized feature and parameter norm bounds imply arm means in
`[-1, 1]`.

This is the finite-action linear-bandit way to discharge the bounded-mean assumption used by the
capped initial-regret term. If `μ(a) = θᵀx_a`, `θᵀθ ≤ S2`, `x_aᵀx_a ≤ L2`, and `L2 * S2 ≤ 1`,
then Cauchy-Schwarz gives `|μ(a)| ≤ √S2 * √L2 ≤ 1`. -/
lemma meanRewardBound_neg_one_one_of_linear_sq_norm_bounds
    {L2 S2 : ℝ} {θ : Feature d}
    (h_linear : LinearMeanModel ν x θ)
    (hL2 : FeatureSqNormBound x L2)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1) :
    MeanRewardBound (K := K) ν (-1) 1 := by
  intro a
  have hS2_nonneg : 0 ≤ S2 := (dotProduct_self_nonneg θ).trans hθ
  have hL2_nonneg : 0 ≤ L2 := (featureSqNorm_nonneg x a).trans (hL2 a)
  have hdot :
      |dotProduct θ (x a)| ≤ √S2 * √L2 :=
    abs_dotProduct_le_sqrt_mul_sqrt_of_sq_norm_le (u := θ) (v := x a)
      (U := S2) (V := L2) hθ (by simpa [featureSqNorm] using hL2 a)
  have hsqrt_prod_le : √S2 * √L2 ≤ 1 := by
    have hprod_le : S2 * L2 ≤ 1 := by
      nlinarith [hLS_le_one]
    calc
      √S2 * √L2 = √(S2 * L2) := by
        rw [Real.sqrt_mul hS2_nonneg]
      _ ≤ √1 := Real.sqrt_le_sqrt hprod_le
      _ = 1 := by norm_num
  have habs : |(ν a)[id]| ≤ 1 := by
    rw [h_linear a]
    exact hdot.trans hsqrt_prod_le
  exact abs_le.mp habs

omit [IsMarkovKernel ν] in
/-- If every arm mean lies in `[lo, hi]`, then every arm gap is at most `hi - lo`. -/
lemma gap_le_of_meanRewardBound [Nonempty (Fin K)] {lo hi : ℝ}
    (hμ : MeanRewardBound ν lo hi) (a : Fin K) :
    gap ν a ≤ hi - lo := by
  rw [gap_eq_bestArm_sub]
  have hbest_le : (ν (bestArm ν))[id] ≤ hi := (hμ (bestArm ν)).2
  have ha_ge : lo ≤ (ν a)[id] := (hμ a).1
  linarith

omit [IsMarkovKernel ν] in
/-- Arm means in `[-1, 1]` imply the gap cap `gap ≤ 2` used by the capped regret argument. -/
lemma gapBound_two_of_meanRewardBound_neg_one_one [Nonempty (Fin K)]
    (hμ : MeanRewardBound ν (-1) 1) :
    GapBound (K := K) ν 2 := by
  intro a
  have hgap := gap_le_of_meanRewardBound (ν := ν) (lo := -1) (hi := 1) hμ a
  norm_num at hgap
  exact hgap

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- The initial-gap term used by this formalization is deterministically at most `2` when all arm
means lie in `[-1, 1]`. At horizon zero the initial term is exactly zero. -/
lemma initialGapTerm_le_two_of_meanRewardBound_neg_one_one [Nonempty (Fin K)]
    (hμ : MeanRewardBound ν (-1) 1) :
    (if n = 0 then 0 else gap ν (A 0 ω)) ≤ if n = 0 then 0 else 2 := by
  by_cases hn : n = 0
  · simp [hn]
  · simpa [hn] using (gapBound_two_of_meanRewardBound_neg_one_one (ν := ν) hμ (A 0 ω))

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- A uniform gap bound implies the selected-action gap bound through any finite horizon. -/
lemma gap_ae_le_of_GapBound (G : ℝ) (hG : GapBound (K := K) ν G) :
    ∀ᵐ ω ∂P, ∀ t, t ∈ range n → gap ν (A t ω) ≤ G :=
  Filter.Eventually.of_forall fun ω t _ht ↦ hG (A t ω)

omit [IsMarkovKernel ν] in
/-- First projection from the packaged LinUCB confidence event: optimism for the best arm. -/
lemma LinUCBConfidenceEvent.best [Nonempty (Fin K)]
    (h_conf : LinUCBConfidenceEvent A R reg β x ν ω) :
    ∀ t, t ≠ 0 →
      (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) t ω := by
  intro t ht
  exact (h_conf t ht).1

omit [IsMarkovKernel ν] in
/-- Second projection from the packaged LinUCB confidence event: validity of the selected arm's
lower confidence inequality. -/
lemma LinUCBConfidenceEvent.arm [Nonempty (Fin K)]
    (h_conf : LinUCBConfidenceEvent A R reg β x ν ω) :
    ∀ t, t ≠ 0 →
      estimatedReward A R reg x (A t ω) t ω -
        √(β (t + 1)) * width A reg x (A t ω) t ω ≤ (ν (A t ω))[id] := by
  intro t ht
  exact (h_conf t ht).2

omit [IsMarkovKernel ν] in
/-- The uniform self-normalized prediction-error event implies the two inequalities packaged in
`LinUCBConfidenceEvent`. -/
lemma LinUCBConfidenceEvent.of_selfNormalized [Nonempty (Fin K)]
    (h_self : LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω) :
    LinUCBConfidenceEvent A R reg β x ν ω := by
  intro t ht
  constructor
  · have h_abs := h_self t ht (bestArm ν)
    have h_lower := (abs_le.mp h_abs).1
    set est := estimatedReward A R reg x (bestArm ν) t ω
    set μ := (ν (bestArm ν))[id]
    set bonus := √(β (t + 1)) * width A reg x (bestArm ν) t ω
    change μ ≤ est + bonus
    linarith
  · have h_abs := h_self t ht (A t ω)
    have h_upper := (abs_le.mp h_abs).2
    set est := estimatedReward A R reg x (A t ω) t ω
    set μ := (ν (A t ω))[id]
    set bonus := √(β (t + 1)) * width A reg x (A t ω) t ω
    change est - bonus ≤ μ
    linarith

omit [IsMarkovKernel ν] in
/-- The horizon-local self-normalized prediction-error event implies the horizon-local LinUCB
confidence event used by the finite-horizon regret proof. -/
lemma LinUCBConfidenceEventUpTo.of_selfNormalized [Nonempty (Fin K)]
    (h_self : LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω) :
    LinUCBConfidenceEventUpTo A R reg β x ν n ω := by
  intro t ht ht0
  constructor
  · have h_abs := h_self t ht ht0 (bestArm ν)
    have h_lower := (abs_le.mp h_abs).1
    set est := estimatedReward A R reg x (bestArm ν) t ω
    set μ := (ν (bestArm ν))[id]
    set bonus := √(β (t + 1)) * width A reg x (bestArm ν) t ω
    change μ ≤ est + bonus
    linarith
  · have h_abs := h_self t ht ht0 (A t ω)
    have h_upper := (abs_le.mp h_abs).2
    set est := estimatedReward A R reg x (A t ω) t ω
    set μ := (ν (A t ω))[id]
    set bonus := √(β (t + 1)) * width A reg x (A t ω) t ω
    change est - bonus ≤ μ
    linarith

omit [IsMarkovKernel ν] in
/-- The horizon-local self-normalized event is contained in the horizon-local LinUCB confidence
event. -/
lemma probReal_selfNormalizedConfidenceEventUpTo_le_confidenceEventUpTo
    [Nonempty (Fin K)] :
    P.real {ω | LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω} ≤
      P.real {ω | LinUCBConfidenceEventUpTo A R reg β x ν n ω} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_selfω ↦
    LinUCBConfidenceEventUpTo.of_selfNormalized (A := A) (R := R) (reg := reg)
      (β := β) (x := x) (ν := ν) (n := n) (ω := ω) h_selfω

omit [IsMarkovKernel ν] in
/-- High-probability bridge from the horizon-local self-normalized prediction-error event to the
horizon-local LinUCB confidence event. -/
lemma probReal_confidenceEventUpTo_ge_of_selfNormalizedConfidenceEventUpTo_ge
    [Nonempty (Fin K)] {δ : ℝ}
    (h_self_prob :
      1 - δ ≤ P.real {ω | LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω}) :
    1 - δ ≤ P.real {ω | LinUCBConfidenceEventUpTo A R reg β x ν n ω} :=
  h_self_prob.trans
    (probReal_selfNormalizedConfidenceEventUpTo_le_confidenceEventUpTo
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      (P := P))

omit [IsMarkovKernel ν] in
/-- Failure-probability bridge from the horizon-local self-normalized prediction-error event to the
horizon-local LinUCB confidence event. -/
lemma probReal_confidenceEventUpTo_failure_le_of_selfNormalizedConfidenceEventUpTo_failure_le
    [Nonempty (Fin K)] {δ : ℝ}
    (h_self_failure :
      P.real {ω | ¬ LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω} ≤ δ) :
    P.real {ω | ¬ LinUCBConfidenceEventUpTo A R reg β x ν n ω} ≤ δ := by
  refine le_trans ?_ h_self_failure
  refine probReal_failure_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_selfω ↦
    LinUCBConfidenceEventUpTo.of_selfNormalized (A := A) (R := R) (reg := reg)
      (β := β) (x := x) (ν := ν) (n := n) (ω := ω) h_selfω

/-- Conservative coordinate-tail concentration route to the horizon-local self-normalized
prediction confidence event.

This is the finite-action, coordinate-union-bound replacement for the textbook vector
self-normalized concentration theorem. It packages the already proved route
coordinate bounds -> centered-noise quadratic bound -> ridge-bias bound -> self-normalized
prediction confidence. -/
lemma probReal_selfNormalizedUpTo_failure_le_of_coord_tail_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
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
    P.real {ω | ¬ LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω} ≤ δ := by
  refine probReal_selfNormalizedConfidenceEventUpTo_failure_le_of_centeredNoiseBias_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
    (P := P) θ h_linear hreg_pos ?_
  refine probReal_centeredNoiseBiasUpTo_failure_le_of_centeredNoiseUpTo_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
    (P := P) (noiseBudget := centeredNoiseCoordinateNoiseBudget d reg coordBudget)
    θ S2 hreg_pos hθ ?_ ?_
  · intro t ht ht0
    simpa [centeredNoiseCoordinateBetaBudget] using hβ_budget t ht ht0
  · refine
      probReal_centeredNoiseConfidenceEventUpTo_failure_le_of_coordinateBoundEventUpTo_failure_le
        (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) (P := P)
        (noiseBudget := centeredNoiseCoordinateNoiseBudget d reg coordBudget)
        hreg_pos hcoord_nonneg ?_ ?_
    · intro t _ht _ht0
      simp [centeredNoiseCoordinateNoiseBudget]
    · exact
        (probReal_centeredNoiseCoordinateBoundEventUpTo_failure_le_tail_sum
          (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
          (P := P) h hν L2 hL2 hcoord_nonneg).trans
          (centeredNoiseCoordinateTailFailureSum_le_of_coord_tail_le (d := d) (n := n)
            (σ2 := σ2) (L2 := L2) (coordBudget := coordBudget) hδ_nonneg hn hd h_tail)

/-- Conservative coordinate-tail concentration route to the horizon-local LinUCB confidence event.

The regret theorem consumes `LinUCBConfidenceEventUpTo`; this lemma lets callers supply a
coordinate-wise tail schedule instead of a pre-built confidence-event failure bound. -/
lemma probReal_confidenceEventUpTo_failure_le_of_coord_tail_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
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
    P.real {ω | ¬ LinUCBConfidenceEventUpTo A R reg β x ν n ω} ≤ δ :=
  probReal_confidenceEventUpTo_failure_le_of_selfNormalizedConfidenceEventUpTo_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (P := P)
    (probReal_selfNormalizedUpTo_failure_le_of_coord_tail_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h hν
      hreg_pos L2 hL2 θ h_linear S2 hθ hcoord_nonneg hβ_budget hδ_nonneg hn hd h_tail)

/-- High-probability coordinate-tail concentration route to the horizon-local self-normalized
prediction confidence event. -/
lemma probReal_selfNormalizedUpTo_ge_of_coord_tail_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
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
    1 - δ ≤ P.real {ω | LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω} :=
  probReal_event_ge_of_failure_le (P := P)
    (probReal_selfNormalizedUpTo_failure_le_of_coord_tail_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h hν
      hreg_pos L2 hL2 θ h_linear S2 hθ hcoord_nonneg hβ_budget hδ_nonneg hn hd h_tail)

/-- High-probability coordinate-tail concentration route to the horizon-local LinUCB confidence
event. -/
lemma probReal_confidenceEventUpTo_ge_of_coord_tail_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
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
    1 - δ ≤ P.real {ω | LinUCBConfidenceEventUpTo A R reg β x ν n ω} :=
  probReal_event_ge_of_failure_le (P := P)
    (probReal_confidenceEventUpTo_failure_le_of_coord_tail_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h hν
      hreg_pos L2 hL2 θ h_linear S2 hθ hcoord_nonneg hβ_budget hδ_nonneg hn hd h_tail)

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost-sure bridge from the self-normalized prediction-error event to the LinUCB confidence
event used by the regret proof. -/
lemma linUCBConfidenceEvent_ae_of_selfNormalizedConfidenceEvent_ae [Nonempty (Fin K)]
    (h_self : ∀ᵐ ω ∂P, LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω) :
    ∀ᵐ ω ∂P, LinUCBConfidenceEvent A R reg β x ν ω := by
  filter_upwards [h_self] with ω h_selfω
  exact LinUCBConfidenceEvent.of_selfNormalized (A := A) (R := R) (reg := reg)
    (β := β) (x := x) (ν := ν) (ω := ω) h_selfω

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost-sure bridge from parameter prediction confidence to self-normalized prediction
confidence under linear realizability. -/
lemma linUCBSelfNormalizedConfidenceEvent_ae_of_parameterPredictionConfidenceEvent_ae
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (h_param : ∀ᵐ ω ∂P, LinUCBParameterPredictionConfidenceEvent A R reg β x θ ω) :
    ∀ᵐ ω ∂P, LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω := by
  filter_upwards [h_param] with ω h_paramω
  exact LinUCBSelfNormalizedConfidenceEvent.of_parameterPrediction (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (ω := ω) θ h_linear h_paramω

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost-sure bridge from parameter prediction confidence directly to the LinUCB confidence event
used by the regret proof. -/
lemma linUCBConfidenceEvent_ae_of_parameterPredictionConfidenceEvent_ae [Nonempty (Fin K)]
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (h_param : ∀ᵐ ω ∂P, LinUCBParameterPredictionConfidenceEvent A R reg β x θ ω) :
    ∀ᵐ ω ∂P, LinUCBConfidenceEvent A R reg β x ν ω := by
  exact linUCBConfidenceEvent_ae_of_selfNormalizedConfidenceEvent_ae (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (P := P)
    (linUCBSelfNormalizedConfidenceEvent_ae_of_parameterPredictionConfidenceEvent_ae
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (P := P)
      θ h_linear h_param)

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost-sure bridge from centered-noise-plus-bias confidence to the textbook parameter
ellipsoid event. -/
lemma linUCBParameterEllipsoidConfidenceEvent_ae_of_centeredNoiseBiasConfidenceEvent_ae
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg)
    (h_noise :
      ∀ᵐ ω ∂P, LinUCBCenteredNoiseBiasConfidenceEvent A R reg β x ν θ ω) :
    ∀ᵐ ω ∂P, LinUCBParameterEllipsoidConfidenceEvent A R reg β x θ ω := by
  filter_upwards [h_noise] with ω h_noiseω
  exact LinUCBParameterEllipsoidConfidenceEvent.of_centeredNoiseBias (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (ω := ω) θ h_linear hreg_pos
    h_noiseω

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost-sure bridge from textbook ellipsoid confidence to parameter prediction confidence under
positive regularization. -/
lemma linUCBParameterPredictionConfidenceEvent_ae_of_parameterEllipsoidConfidenceEvent_ae
    (θ : Feature d)
    (hreg_pos : 0 < reg)
    (h_ellipsoid : ∀ᵐ ω ∂P, LinUCBParameterEllipsoidConfidenceEvent A R reg β x θ ω) :
    ∀ᵐ ω ∂P, LinUCBParameterPredictionConfidenceEvent A R reg β x θ ω := by
  filter_upwards [h_ellipsoid] with ω h_ellipsoidω
  exact LinUCBParameterPredictionConfidenceEvent.of_ellipsoid (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ω := ω) θ hreg_pos h_ellipsoidω

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost-sure bridge from textbook ellipsoid confidence to self-normalized prediction confidence
under positive regularization and linear realizability. -/
lemma linUCBSelfNormalizedConfidenceEvent_ae_of_parameterEllipsoidConfidenceEvent_ae
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg)
    (h_ellipsoid : ∀ᵐ ω ∂P, LinUCBParameterEllipsoidConfidenceEvent A R reg β x θ ω) :
    ∀ᵐ ω ∂P, LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω := by
  filter_upwards [h_ellipsoid] with ω h_ellipsoidω
  exact LinUCBSelfNormalizedConfidenceEvent.of_parameterEllipsoid (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (ω := ω) θ h_linear hreg_pos
    h_ellipsoidω

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost-sure bridge from textbook ellipsoid confidence directly to the LinUCB confidence event
used by the regret proof. -/
lemma linUCBConfidenceEvent_ae_of_parameterEllipsoidConfidenceEvent_ae [Nonempty (Fin K)]
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg)
    (h_ellipsoid : ∀ᵐ ω ∂P, LinUCBParameterEllipsoidConfidenceEvent A R reg β x θ ω) :
    ∀ᵐ ω ∂P, LinUCBConfidenceEvent A R reg β x ν ω := by
  exact linUCBConfidenceEvent_ae_of_selfNormalizedConfidenceEvent_ae (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (P := P)
    (linUCBSelfNormalizedConfidenceEvent_ae_of_parameterEllipsoidConfidenceEvent_ae
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (P := P)
      θ h_linear hreg_pos h_ellipsoid)

omit [IsMarkovKernel ν] in
/-- The self-normalized prediction-error event is contained in `LinUCBConfidenceEvent`, hence its
probability is a lower bound on the probability of the LinUCB confidence event. -/
lemma probReal_selfNormalizedConfidenceEvent_le_confidenceEvent [Nonempty (Fin K)] :
    P.real {ω | LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω} ≤
      P.real {ω | LinUCBConfidenceEvent A R reg β x ν ω} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_selfω ↦
    LinUCBConfidenceEvent.of_selfNormalized (A := A) (R := R) (reg := reg)
      (β := β) (x := x) (ν := ν) (ω := ω) h_selfω

omit [IsMarkovKernel ν] in
/-- A parameter prediction-confidence event is contained in the self-normalized confidence event
under linear realizability. -/
lemma probReal_parameterPredictionConfidenceEvent_le_selfNormalizedConfidenceEvent
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ) :
    P.real {ω | LinUCBParameterPredictionConfidenceEvent A R reg β x θ ω} ≤
      P.real {ω | LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_paramω ↦
    LinUCBSelfNormalizedConfidenceEvent.of_parameterPrediction (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (ω := ω) θ h_linear h_paramω

omit [IsMarkovKernel ν] in
/-- A centered-noise-plus-bias confidence event is contained in the textbook parameter ellipsoid
event under linear realizability and positive regularization. -/
lemma probReal_centeredNoiseBiasConfidenceEvent_le_parameterEllipsoidConfidenceEvent
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg) :
    P.real {ω | LinUCBCenteredNoiseBiasConfidenceEvent A R reg β x ν θ ω} ≤
      P.real {ω | LinUCBParameterEllipsoidConfidenceEvent A R reg β x θ ω} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_noiseω ↦
    LinUCBParameterEllipsoidConfidenceEvent.of_centeredNoiseBias (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (ω := ω) θ h_linear hreg_pos
      h_noiseω

omit [IsMarkovKernel ν] in
/-- A textbook ellipsoid confidence event is contained in the parameter prediction-confidence event
under positive regularization. -/
lemma probReal_parameterEllipsoidConfidenceEvent_le_parameterPredictionConfidenceEvent
    (θ : Feature d)
    (hreg_pos : 0 < reg) :
    P.real {ω | LinUCBParameterEllipsoidConfidenceEvent A R reg β x θ ω} ≤
      P.real {ω | LinUCBParameterPredictionConfidenceEvent A R reg β x θ ω} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_ellipsoidω ↦
    LinUCBParameterPredictionConfidenceEvent.of_ellipsoid (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ω := ω) θ hreg_pos h_ellipsoidω

omit [IsMarkovKernel ν] in
/-- A textbook ellipsoid confidence event is contained in the self-normalized confidence event under
positive regularization and linear realizability. -/
lemma probReal_parameterEllipsoidConfidenceEvent_le_selfNormalizedConfidenceEvent
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg) :
    P.real {ω | LinUCBParameterEllipsoidConfidenceEvent A R reg β x θ ω} ≤
      P.real {ω | LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω} := by
  exact (probReal_parameterEllipsoidConfidenceEvent_le_parameterPredictionConfidenceEvent
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (P := P) θ hreg_pos).trans
    (probReal_parameterPredictionConfidenceEvent_le_selfNormalizedConfidenceEvent
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (P := P)
      θ h_linear)

omit [IsMarkovKernel ν] in
/-- High-probability bridge from centered-noise-plus-bias confidence to the textbook parameter
ellipsoid event. -/
lemma probReal_parameterEllipsoidConfidenceEvent_ge_of_centeredNoiseBiasConfidenceEvent_ge
    (θ : Feature d) {δ : ℝ}
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg)
    (h_noise_prob :
      1 - δ ≤ P.real {ω | LinUCBCenteredNoiseBiasConfidenceEvent A R reg β x ν θ ω}) :
    1 - δ ≤ P.real {ω | LinUCBParameterEllipsoidConfidenceEvent A R reg β x θ ω} :=
  h_noise_prob.trans
    (probReal_centeredNoiseBiasConfidenceEvent_le_parameterEllipsoidConfidenceEvent
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (P := P)
      θ h_linear hreg_pos)

omit [IsMarkovKernel ν] in
/-- High-probability bridge from parameter prediction confidence to self-normalized prediction
confidence under linear realizability. -/
lemma probReal_selfNormalizedConfidenceEvent_ge_of_parameterPredictionConfidenceEvent_ge
    (θ : Feature d) {δ : ℝ}
    (h_linear : LinearMeanModel ν x θ)
    (h_param_prob :
      1 - δ ≤ P.real {ω | LinUCBParameterPredictionConfidenceEvent A R reg β x θ ω}) :
    1 - δ ≤ P.real {ω | LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω} :=
  h_param_prob.trans
    (probReal_parameterPredictionConfidenceEvent_le_selfNormalizedConfidenceEvent
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (P := P)
      θ h_linear)

omit [IsMarkovKernel ν] in
/-- High-probability bridge from textbook ellipsoid confidence to self-normalized prediction
confidence under linear realizability. -/
lemma probReal_selfNormalizedConfidenceEvent_ge_of_parameterEllipsoidConfidenceEvent_ge
    (θ : Feature d) {δ : ℝ}
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg)
    (h_ellipsoid_prob :
      1 - δ ≤ P.real {ω | LinUCBParameterEllipsoidConfidenceEvent A R reg β x θ ω}) :
    1 - δ ≤ P.real {ω | LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω} :=
  h_ellipsoid_prob.trans
    (probReal_parameterEllipsoidConfidenceEvent_le_selfNormalizedConfidenceEvent
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (P := P)
      θ h_linear hreg_pos)

omit [IsMarkovKernel ν] in
/-- Failure-probability bridge from parameter prediction confidence to self-normalized prediction
confidence under linear realizability. -/
lemma probReal_selfNormalized_failure_le_of_parameterPrediction_failure_le
    (θ : Feature d) {δ : ℝ}
    (h_linear : LinearMeanModel ν x θ)
    (h_param_failure :
      P.real {ω | ¬ LinUCBParameterPredictionConfidenceEvent A R reg β x θ ω} ≤ δ) :
    P.real {ω | ¬ LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω} ≤ δ := by
  refine le_trans ?_ h_param_failure
  refine probReal_failure_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_paramω ↦
    LinUCBSelfNormalizedConfidenceEvent.of_parameterPrediction (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (ω := ω) θ h_linear h_paramω

omit [IsMarkovKernel ν] in
/-- Failure-probability bridge from centered-noise-plus-bias confidence to the textbook parameter
ellipsoid event. -/
lemma probReal_parameterEllipsoid_failure_le_of_centeredNoiseBias_failure_le
    (θ : Feature d) {δ : ℝ}
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg)
    (h_noise_failure :
      P.real {ω | ¬ LinUCBCenteredNoiseBiasConfidenceEvent A R reg β x ν θ ω} ≤ δ) :
    P.real {ω | ¬ LinUCBParameterEllipsoidConfidenceEvent A R reg β x θ ω} ≤ δ := by
  refine le_trans ?_ h_noise_failure
  refine probReal_failure_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_noiseω ↦
    LinUCBParameterEllipsoidConfidenceEvent.of_centeredNoiseBias (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (ω := ω) θ h_linear hreg_pos
      h_noiseω

omit [IsMarkovKernel ν] in
/-- Failure-probability bridge from textbook ellipsoid confidence to self-normalized prediction
confidence under linear realizability. -/
lemma probReal_selfNormalized_failure_le_of_parameterEllipsoid_failure_le
    (θ : Feature d) {δ : ℝ}
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg)
    (h_ellipsoid_failure :
      P.real {ω | ¬ LinUCBParameterEllipsoidConfidenceEvent A R reg β x θ ω} ≤ δ) :
    P.real {ω | ¬ LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω} ≤ δ := by
  refine le_trans ?_ h_ellipsoid_failure
  refine probReal_failure_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_ellipsoidω ↦
    LinUCBSelfNormalizedConfidenceEvent.of_parameterEllipsoid (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (ω := ω) θ h_linear hreg_pos
      h_ellipsoidω

omit [IsMarkovKernel ν] in
/-- High-probability bridge from a self-normalized prediction-error event to
`LinUCBConfidenceEvent`. -/
lemma probReal_confidenceEvent_ge_of_selfNormalizedConfidenceEvent_ge [Nonempty (Fin K)] {δ : ℝ}
    (h_self_prob :
      1 - δ ≤ P.real {ω | LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω}) :
    1 - δ ≤ P.real {ω | LinUCBConfidenceEvent A R reg β x ν ω} :=
  h_self_prob.trans
    (probReal_selfNormalizedConfidenceEvent_le_confidenceEvent (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (P := P))

omit [IsMarkovKernel ν] in
/-- Failure-probability bridge from the self-normalized prediction-error event to
`LinUCBConfidenceEvent`. -/
lemma probReal_confidenceEvent_failure_le_of_selfNormalizedConfidenceEvent_failure_le
    [Nonempty (Fin K)] {δ : ℝ}
    (h_self_failure :
      P.real {ω | ¬ LinUCBSelfNormalizedConfidenceEvent A R reg β x ν ω} ≤ δ) :
    P.real {ω | ¬ LinUCBConfidenceEvent A R reg β x ν ω} ≤ δ := by
  refine le_trans ?_ h_self_failure
  refine probReal_failure_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_selfω ↦
    LinUCBConfidenceEvent.of_selfNormalized (A := A) (R := R) (reg := reg)
      (β := β) (x := x) (ν := ν) (ω := ω) h_selfω

omit [IsMarkovKernel ν] in
/-- In zero feature dimension, the confidence event forces every positive-time selected gap to be
nonpositive. The best-arm index is zero, and the selected-arm pessimistic index is also zero. -/
lemma gap_nonpos_of_confidence_dim_eq_zero [Nonempty (Fin K)]
    (hd : d = 0) (h_conf : LinUCBConfidenceEvent A R reg β x ν ω)
    (t : ℕ) (ht : t ≠ 0) :
    gap ν (A t ω) ≤ 0 := by
  have hbest := LinUCBConfidenceEvent.best (A := A) (R := R) (reg := reg) (β := β)
    (x := x) (ν := ν) (ω := ω) h_conf t ht
  have harm := LinUCBConfidenceEvent.arm (A := A) (R := R) (reg := reg) (β := β)
    (x := x) (ν := ν) (ω := ω) h_conf t ht
  rw [gap_eq_bestArm_sub]
  have hbest0 : (ν (bestArm ν))[id] ≤ 0 := by
    simpa [index_eq_zero_of_dim_eq_zero (A := A) (R := R) (reg := reg) (β := β)
      (x := x) (n := t) (ω := ω) hd (bestArm ν)] using hbest
  have harm0 : 0 ≤ (ν (A t ω))[id] := by
    simpa [estimatedReward_eq_zero_of_dim_eq_zero (A := A) (R := R) (reg := reg)
      (x := x) (n := t) (ω := ω) hd (A t ω),
      width_eq_zero_of_dim_eq_zero (A := A) (reg := reg) (x := x) (n := t)
        (ω := ω) hd (A t ω)] using harm
  linarith

omit [IsMarkovKernel ν] in
/-- Horizon-local zero-dimensional version: the confidence event up to `n` forces every selected
positive-time gap in `range n` to be nonpositive. -/
lemma gap_nonpos_of_confidenceUpTo_dim_eq_zero [Nonempty (Fin K)]
    (hd : d = 0) (h_conf : LinUCBConfidenceEventUpTo A R reg β x ν n ω)
    (t : ℕ) (ht : t ∈ range n) (ht0 : t ≠ 0) :
    gap ν (A t ω) ≤ 0 := by
  have hbest := LinUCBConfidenceEventUpTo.best (A := A) (R := R) (reg := reg)
    (β := β) (x := x) (ν := ν) (n := n) (ω := ω) h_conf t ht ht0
  have harm := LinUCBConfidenceEventUpTo.arm (A := A) (R := R) (reg := reg)
    (β := β) (x := x) (ν := ν) (n := n) (ω := ω) h_conf t ht ht0
  rw [gap_eq_bestArm_sub]
  have hbest0 : (ν (bestArm ν))[id] ≤ 0 := by
    simpa [index_eq_zero_of_dim_eq_zero (A := A) (R := R) (reg := reg) (β := β)
      (x := x) (n := t) (ω := ω) hd (bestArm ν)] using hbest
  have harm0 : 0 ≤ (ν (A t ω))[id] := by
    simpa [estimatedReward_eq_zero_of_dim_eq_zero (A := A) (R := R) (reg := reg)
      (x := x) (n := t) (ω := ω) hd (A t ω),
      width_eq_zero_of_dim_eq_zero (A := A) (reg := reg) (x := x) (n := t)
        (ω := ω) hd (A t ω)] using harm
  linarith

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost-sure projection of the packaged confidence event to optimism for the best arm. -/
lemma linUCBConfidenceEvent_ae_best [Nonempty (Fin K)]
    (h_conf : ∀ᵐ ω ∂P, LinUCBConfidenceEvent A R reg β x ν ω) :
    ∀ᵐ ω ∂P, ∀ t, t ≠ 0 →
      (ν (bestArm ν))[id] ≤ index A R reg β x (bestArm ν) t ω := by
  filter_upwards [h_conf] with ω h_confω
  exact h_confω.best

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost-sure projection of the packaged confidence event to the selected arm's lower confidence
inequality. -/
lemma linUCBConfidenceEvent_ae_arm [Nonempty (Fin K)]
    (h_conf : ∀ᵐ ω ∂P, LinUCBConfidenceEvent A R reg β x ν ω) :
    ∀ᵐ ω ∂P, ∀ t, t ≠ 0 →
      estimatedReward A R reg x (A t ω) t ω -
        √(β (t + 1)) * width A reg x (A t ω) t ω ≤ (ν (A t ω))[id] := by
  filter_upwards [h_conf] with ω h_confω
  exact h_confω.arm

lemma designMatrix_eq_designMatrix' (reg : ℝ) (x : Fin K → Feature d) (n : ℕ)
    (ω : Ω) (hn : n ≠ 0) :
    designMatrix A reg x n ω =
      designMatrix' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) := by
  cases n with
  | zero => exact absurd rfl hn
  | succ n =>
    simp only [designMatrix, designMatrix', IsAlgEnvSeq.hist]
    rw [Nat.range_succ_eq_Iic]
    exact congrArg (fun S ↦ reg • 1 + S) <|
      (Finset.sum_coe_sort (Iic n)
        (fun s ↦ Matrix.vecMulVec (x (A s ω)) (x (A s ω)))).symm

lemma responseVector_eq_responseVector' (x : Fin K → Feature d)
    (n : ℕ) (ω : Ω) (hn : n ≠ 0) :
    responseVector A R x n ω = responseVector' x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) := by
  cases n with
  | zero => exact absurd rfl hn
  | succ n =>
    simp only [responseVector, responseVector', IsAlgEnvSeq.hist]
    rw [Nat.range_succ_eq_Iic]
    exact (Finset.sum_coe_sort (Iic n) (fun s ↦ R s ω • x (A s ω))).symm

lemma thetaHat_eq_thetaHat' (reg : ℝ) (x : Fin K → Feature d)
    (n : ℕ) (ω : Ω) (hn : n ≠ 0) :
    thetaHat A R reg x n ω = thetaHat' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) := by
  simp [thetaHat, thetaHat', designMatrix_eq_designMatrix' (A := A) (R := R) reg x n ω hn,
    responseVector_eq_responseVector' (A := A) (R := R) x n ω hn]

lemma estimatedReward_eq_estimatedReward' (reg : ℝ) (x : Fin K → Feature d)
    (a : Fin K) (n : ℕ) (ω : Ω) (hn : n ≠ 0) :
    estimatedReward A R reg x a n ω =
      estimatedReward' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) a := by
  simp [estimatedReward, estimatedReward', thetaHat_eq_thetaHat' (A := A) (R := R) reg x n ω hn]

lemma widthQuadraticForm_eq_widthQuadraticForm' (reg : ℝ) (x : Fin K → Feature d)
    (a : Fin K) (n : ℕ) (ω : Ω) (hn : n ≠ 0) :
    widthQuadraticForm A reg x a n ω =
      widthQuadraticForm' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) a := by
  simp [widthQuadraticForm, widthQuadraticForm',
    designMatrix_eq_designMatrix' (A := A) (R := R) reg x n ω hn]

/-- At positive process times, nonnegativity of the process-level width quadratic form is
equivalent to nonnegativity of the matching history-level width quadratic form. -/
lemma widthQuadraticForm_nonneg_iff_widthQuadraticForm' (reg : ℝ) (x : Fin K → Feature d)
    (a : Fin K) (n : ℕ) (ω : Ω) (hn : n ≠ 0) :
    0 ≤ widthQuadraticForm A reg x a n ω ↔
      0 ≤ widthQuadraticForm' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) a := by
  rw [widthQuadraticForm_eq_widthQuadraticForm' (A := A) (R := R) reg x a n ω hn]

/-- At positive process times, the process-level quadratic width form is at most `1` iff the
matching history-level quadratic width form is at most `1`. -/
lemma widthQuadraticForm_le_one_iff_widthQuadraticForm' (reg : ℝ) (x : Fin K → Feature d)
    (a : Fin K) (n : ℕ) (ω : Ω) (hn : n ≠ 0) :
    widthQuadraticForm A reg x a n ω ≤ 1 ↔
      widthQuadraticForm' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) a ≤ 1 := by
  rw [widthQuadraticForm_eq_widthQuadraticForm' (A := A) (R := R) reg x a n ω hn]

/-- The all-positive-times process-level nonnegativity assumption is equivalent to the matching
history-level nonnegativity assumption. -/
lemma widthQuadraticForm_all_nonneg_iff_history (reg : ℝ) (x : Fin K → Feature d)
    (n : ℕ) (ω : Ω) :
    (∀ t, t ∈ range n → t ≠ 0 → 0 ≤ widthQuadraticForm A reg x (A t ω) t ω) ↔
      ∀ t, t ∈ range n → t ≠ 0 →
        0 ≤ widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω) := by
  constructor
  · intro h t ht ht0
    exact (widthQuadraticForm_nonneg_iff_widthQuadraticForm' (A := A) (R := R) reg x
      (A t ω) t ω ht0).1 (h t ht ht0)
  · intro h t ht ht0
    exact (widthQuadraticForm_nonneg_iff_widthQuadraticForm' (A := A) (R := R) reg x
      (A t ω) t ω ht0).2 (h t ht ht0)

/-- The all-positive-times process-level `≤ 1` assumption is equivalent to the matching
history-level `≤ 1` assumption. -/
lemma widthQuadraticForm_all_le_one_iff_history (reg : ℝ) (x : Fin K → Feature d)
    (n : ℕ) (ω : Ω) :
    (∀ t, t ∈ range n → t ≠ 0 → widthQuadraticForm A reg x (A t ω) t ω ≤ 1) ↔
      ∀ t, t ∈ range n → t ≠ 0 →
        widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω) ≤ 1 := by
  constructor
  · intro h t ht ht0
    exact (widthQuadraticForm_le_one_iff_widthQuadraticForm' (A := A) (R := R) reg x
      (A t ω) t ω ht0).1 (h t ht ht0)
  · intro h t ht ht0
    exact (widthQuadraticForm_le_one_iff_widthQuadraticForm' (A := A) (R := R) reg x
      (A t ω) t ω ht0).2 (h t ht ht0)

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost surely, process-level all-positive-times nonnegativity is equivalent to the matching
history-level nonnegativity assumption. -/
lemma widthQuadraticForm_ae_all_nonneg_iff_history (reg : ℝ) (x : Fin K → Feature d)
    (n : ℕ) :
    (∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm A reg x (A t ω) t ω) ↔
      ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
        0 ≤ widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω) := by
  constructor
  · intro h
    filter_upwards [h] with ω hω
    exact (widthQuadraticForm_all_nonneg_iff_history (A := A) (R := R) reg x n ω).1 hω
  · intro h
    filter_upwards [h] with ω hω
    exact (widthQuadraticForm_all_nonneg_iff_history (A := A) (R := R) reg x n ω).2 hω

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost surely, the process-level all-positive-times `≤ 1` assumption is equivalent to the
matching history-level `≤ 1` assumption. -/
lemma widthQuadraticForm_ae_all_le_one_iff_history (reg : ℝ) (x : Fin K → Feature d)
    (n : ℕ) :
    (∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      widthQuadraticForm A reg x (A t ω) t ω ≤ 1) ↔
      ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
        widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω) ≤ 1 := by
  constructor
  · intro h
    filter_upwards [h] with ω hω
    exact (widthQuadraticForm_all_le_one_iff_history (A := A) (R := R) reg x n ω).1 hω
  · intro h
    filter_upwards [h] with ω hω
    exact (widthQuadraticForm_all_le_one_iff_history (A := A) (R := R) reg x n ω).2 hω

lemma width_eq_width' (reg : ℝ) (x : Fin K → Feature d)
    (a : Fin K) (n : ℕ) (ω : Ω) (hn : n ≠ 0) :
    width A reg x a n ω = width' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) a := by
  simp [width, width', widthQuadraticForm_eq_widthQuadraticForm' (A := A) (R := R) reg x a n
    ω hn]

/-- At positive process times, squaring the process-level width recovers the matching history-level
quadratic form when that history-level quadratic form is nonnegative. -/
lemma width_sq_eq_widthQuadraticForm' (reg : ℝ) (x : Fin K → Feature d)
    (a : Fin K) (n : ℕ) (ω : Ω) (hn : n ≠ 0)
    (h_nonneg :
      0 ≤ widthQuadraticForm' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) a) :
    width A reg x a n ω ^ 2 =
      widthQuadraticForm' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) a := by
  rw [width_eq_width' (A := A) (R := R) reg x a n ω hn]
  exact width'_sq_eq_quadratic_form reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) a
    h_nonneg

/-- At positive process times, advancing `widthSqSum` adds the matching history-level quadratic
form when that history-level quadratic form is nonnegative. -/
lemma widthSqSum_succ_eq_add_widthQuadraticForm' (reg : ℝ) (x : Fin K → Feature d)
    (n : ℕ) (ω : Ω) (hn : n ≠ 0)
    (h_nonneg :
      0 ≤ widthQuadraticForm' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) (A n ω)) :
    widthSqSum A reg x (n + 1) ω =
      widthSqSum A reg x n ω +
        widthQuadraticForm' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) (A n ω) := by
  rw [widthSqSum_succ_of_ne_zero (A := A) (reg := reg) (x := x) (n := n) (ω := ω) hn]
  rw [width_sq_eq_widthQuadraticForm' (A := A) (R := R) reg x (A n ω) n ω hn h_nonneg]

/-- At positive process times, advancing `quadraticWidthSum` adds the matching history-level
quadratic form. -/
lemma quadraticWidthSum_succ_eq_add_widthQuadraticForm' (reg : ℝ) (x : Fin K → Feature d)
    (n : ℕ) (ω : Ω) (hn : n ≠ 0) :
    quadraticWidthSum A reg x (n + 1) ω =
      quadraticWidthSum A reg x n ω +
        widthQuadraticForm' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) (A n ω) := by
  rw [quadraticWidthSum_succ_of_ne_zero (A := A) (reg := reg) (x := x) (n := n)
    (ω := ω) hn]
  rw [widthQuadraticForm_eq_widthQuadraticForm' (A := A) (R := R) reg x (A n ω) n ω hn]

/-- The history-level quadratic-form accumulator aligned with process times.

The term at process time `t = 0` is set to zero, matching the convention used by `widthSqSum` and
`quadraticWidthSum`. At positive process time `t`, the history available to LinUCB is
`IsAlgEnvSeq.hist A R (t - 1) ω`. -/
noncomputable def historyQuadraticWidthSum (A : ℕ → Ω → Fin K) (R : ℕ → Ω → ℝ)
    (reg : ℝ) (x : Fin K → Feature d) (n : ℕ) (ω : Ω) : ℝ :=
  ∑ t ∈ range n,
    if t = 0 then 0 else
      widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω)

/-- No positive-time history-level quadratic width forms are accumulated at horizon zero. -/
lemma historyQuadraticWidthSum_zero (A : ℕ → Ω → Fin K) (R : ℕ → Ω → ℝ)
    (reg : ℝ) (x : Fin K → Feature d) (ω : Ω) :
    historyQuadraticWidthSum A R reg x 0 ω = 0 := by
  simp [historyQuadraticWidthSum]

/-- Advancing the horizon adds the next positive-time history-level quadratic width form. -/
lemma historyQuadraticWidthSum_succ (A : ℕ → Ω → Fin K) (R : ℕ → Ω → ℝ)
    (reg : ℝ) (x : Fin K → Feature d) (n : ℕ) (ω : Ω) :
    historyQuadraticWidthSum A R reg x (n + 1) ω =
      historyQuadraticWidthSum A R reg x n ω +
        if n = 0 then 0 else
          widthQuadraticForm' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) (A n ω) := by
  simp [historyQuadraticWidthSum, sum_range_succ]

/-- At positive process times, advancing the history-level quadratic accumulator adds the selected
arm's history-level quadratic width form. -/
lemma historyQuadraticWidthSum_succ_of_ne_zero (A : ℕ → Ω → Fin K) (R : ℕ → Ω → ℝ)
    (reg : ℝ) (x : Fin K → Feature d) (n : ℕ) (ω : Ω) (hn : n ≠ 0) :
    historyQuadraticWidthSum A R reg x (n + 1) ω =
      historyQuadraticWidthSum A R reg x n ω +
        widthQuadraticForm' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) (A n ω) := by
  simp [historyQuadraticWidthSum_succ, hn]

/-- The capped history-level quadratic-form accumulator aligned with process times.

This is the accumulator shape that commonly appears in elliptical-potential statements:
each positive-time quadratic width form is capped at `1`. -/
noncomputable def historyCappedQuadraticWidthSum (A : ℕ → Ω → Fin K) (R : ℕ → Ω → ℝ)
    (reg : ℝ) (x : Fin K → Feature d) (n : ℕ) (ω : Ω) : ℝ :=
  ∑ t ∈ range n,
    if t = 0 then 0 else
      min 1 (widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω))

/-- No positive-time capped history-level quadratic width forms are accumulated at horizon zero. -/
lemma historyCappedQuadraticWidthSum_zero (A : ℕ → Ω → Fin K) (R : ℕ → Ω → ℝ)
    (reg : ℝ) (x : Fin K → Feature d) (ω : Ω) :
    historyCappedQuadraticWidthSum A R reg x 0 ω = 0 := by
  simp [historyCappedQuadraticWidthSum]

/-- Advancing the horizon adds the next positive-time capped history-level quadratic width form. -/
lemma historyCappedQuadraticWidthSum_succ (A : ℕ → Ω → Fin K) (R : ℕ → Ω → ℝ)
    (reg : ℝ) (x : Fin K → Feature d) (n : ℕ) (ω : Ω) :
    historyCappedQuadraticWidthSum A R reg x (n + 1) ω =
      historyCappedQuadraticWidthSum A R reg x n ω +
        if n = 0 then 0 else
          min 1
            (widthQuadraticForm' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) (A n ω)) := by
  simp [historyCappedQuadraticWidthSum, sum_range_succ]

/-- At positive process times, advancing the capped history-level quadratic accumulator adds the
selected arm's capped history-level quadratic width form. -/
lemma historyCappedQuadraticWidthSum_succ_of_ne_zero
    (A : ℕ → Ω → Fin K) (R : ℕ → Ω → ℝ)
    (reg : ℝ) (x : Fin K → Feature d) (n : ℕ) (ω : Ω) (hn : n ≠ 0) :
    historyCappedQuadraticWidthSum A R reg x (n + 1) ω =
      historyCappedQuadraticWidthSum A R reg x n ω +
        min 1
          (widthQuadraticForm' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) (A n ω)) := by
  simp [historyCappedQuadraticWidthSum_succ, hn]

/-- The process-level capped quadratic-width accumulator equals the history-level capped
accumulator aligned with the same process times. -/
lemma cappedQuadraticWidthSum_eq_historyCappedQuadraticWidthSum (reg : ℝ)
    (x : Fin K → Feature d) (n : ℕ) (ω : Ω) :
    cappedQuadraticWidthSum A reg x n ω =
      historyCappedQuadraticWidthSum A R reg x n ω := by
  rw [cappedQuadraticWidthSum, historyCappedQuadraticWidthSum]
  refine sum_congr rfl ?_
  intro t ht
  by_cases ht0 : t = 0
  · simp [ht0]
  · rw [if_neg ht0, if_neg ht0]
    exact congrArg (fun q : ℝ ↦ min 1 q)
      (widthQuadraticForm_eq_widthQuadraticForm' (A := A) (R := R) reg x (A t ω) t ω ht0)

/-- A process-level capped quadratic-width sum bound is equivalent to the matching history-level
capped quadratic-width sum bound. -/
lemma cappedQuadraticWidthSum_le_iff_historyCappedQuadraticWidthSum_le
    (reg : ℝ) (x : Fin K → Feature d) (n : ℕ) (ω : Ω) (W : ℝ) :
    cappedQuadraticWidthSum A reg x n ω ≤ W ↔
      historyCappedQuadraticWidthSum A R reg x n ω ≤ W := by
  rw [cappedQuadraticWidthSum_eq_historyCappedQuadraticWidthSum (A := A) (R := R)
    reg x n ω]

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost surely, a process-level capped quadratic-width sum bound is equivalent to the matching
history-level capped quadratic-width sum bound. -/
lemma cappedQuadraticWidthSum_ae_le_iff_historyCappedQuadraticWidthSum_ae_le
    (reg : ℝ) (x : Fin K → Feature d) (n : ℕ) (W : ℝ) :
    (∀ᵐ ω ∂P, cappedQuadraticWidthSum A reg x n ω ≤ W) ↔
      ∀ᵐ ω ∂P, historyCappedQuadraticWidthSum A R reg x n ω ≤ W := by
  constructor
  · intro h
    filter_upwards [h] with ω hω
    exact (cappedQuadraticWidthSum_le_iff_historyCappedQuadraticWidthSum_le
      (A := A) (R := R) reg x n ω W).1 hω
  · intro h
    filter_upwards [h] with ω hω
    exact (cappedQuadraticWidthSum_le_iff_historyCappedQuadraticWidthSum_le
      (A := A) (R := R) reg x n ω W).2 hω

/-- If every positive-time history-level quadratic width form is at most `1`, then the uncapped and
capped history-level accumulators agree. -/
lemma historyQuadraticWidthSum_eq_historyCappedQuadraticWidthSum
    (h_le_one : ∀ t, t ∈ range n → t ≠ 0 →
      widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω) ≤ 1) :
    historyQuadraticWidthSum A R reg x n ω =
      historyCappedQuadraticWidthSum A R reg x n ω := by
  rw [historyQuadraticWidthSum, historyCappedQuadraticWidthSum]
  refine sum_congr rfl ?_
  intro t ht
  by_cases ht0 : t = 0
  · simp [ht0]
  · rw [if_neg ht0, if_neg ht0]
    exact (min_eq_right (h_le_one t ht ht0)).symm

/-- The process-level quadratic-width accumulator equals the history-level accumulator aligned with
the same process times. -/
lemma quadraticWidthSum_eq_historyQuadraticWidthSum (reg : ℝ) (x : Fin K → Feature d)
    (n : ℕ) (ω : Ω) :
    quadraticWidthSum A reg x n ω = historyQuadraticWidthSum A R reg x n ω := by
  rw [quadraticWidthSum, historyQuadraticWidthSum]
  refine sum_congr rfl ?_
  intro t ht
  by_cases ht0 : t = 0
  · simp [ht0]
  · rw [if_neg ht0, if_neg ht0]
    exact widthQuadraticForm_eq_widthQuadraticForm' (A := A) (R := R) reg x (A t ω) t ω ht0

/-- The squared-width accumulator equals the history-level quadratic-form accumulator whenever the
positive-time history-level quadratic forms are nonnegative. -/
lemma widthSqSum_eq_historyQuadraticWidthSum
    (h_nonneg : ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω)) :
    widthSqSum A reg x n ω = historyQuadraticWidthSum A R reg x n ω := by
  have h_process_nonneg : ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm A reg x (A t ω) t ω := by
    intro t ht ht0
    exact (widthQuadraticForm_nonneg_iff_widthQuadraticForm' (A := A) (R := R) reg x
      (A t ω) t ω ht0).2 (h_nonneg t ht ht0)
  rw [widthSqSum_eq_sum_quadratic_form (A := A) (reg := reg) (x := x)
    (n := n) (ω := ω) h_process_nonneg]
  exact quadraticWidthSum_eq_historyQuadraticWidthSum (A := A) (R := R) reg x n ω

/-- A bound on the history-level quadratic-form accumulator implies the corresponding bound on
`widthSqSum`, provided the positive-time history-level quadratic forms are nonnegative. -/
lemma widthSqSum_le_of_history_quadratic_width_sum_le {W : ℝ}
    (h_nonneg : ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω))
    (h_hist_le : historyQuadraticWidthSum A R reg x n ω ≤ W) :
    widthSqSum A reg x n ω ≤ W := by
  rw [widthSqSum_eq_historyQuadraticWidthSum (A := A) (R := R) (reg := reg) (x := x)
    (n := n) (ω := ω) h_nonneg]
  exact h_hist_le

omit [IsProbabilityMeasure P] in
/-- Almost surely, a history-level quadratic-form bound gives the `widthSqSum` bound consumed by
the regret chain. -/
lemma widthSqSum_ae_le_of_history_quadratic_width_sum_ae_le {W : ℝ}
    (h_nonneg : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω))
    (h_hist_le : ∀ᵐ ω ∂P, historyQuadraticWidthSum A R reg x n ω ≤ W) :
    ∀ᵐ ω ∂P, widthSqSum A reg x n ω ≤ W := by
  filter_upwards [h_nonneg, h_hist_le] with ω h_nonnegω h_hist_leω
  exact widthSqSum_le_of_history_quadratic_width_sum_le (A := A) (R := R) (reg := reg)
    (x := x) (n := n) (ω := ω) h_nonnegω h_hist_leω

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- The pointwise input expected from a history-level elliptical-potential argument.

It packages the two facts needed to turn a history-level quadratic-width estimate into the
`widthSqSum` estimate used by the regret chain:

* each positive-time quadratic width form is nonnegative;
* their history-level accumulated sum is bounded by `W`. -/
def HistoryQuadraticWidthBound (A : ℕ → Ω → Fin K) (R : ℕ → Ω → ℝ)
    (reg : ℝ) (x : Fin K → Feature d) (n : ℕ) (ω : Ω) (W : ℝ) : Prop :=
  (∀ t, t ∈ range n → t ≠ 0 →
    0 ≤ widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω)) ∧
    historyQuadraticWidthSum A R reg x n ω ≤ W

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Build the packaged history-level quadratic-width input from its two component facts. -/
lemma historyQuadraticWidthBound_of_nonneg_and_sum_le {W : ℝ}
    (h_nonneg : ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω))
    (h_sum_le : historyQuadraticWidthSum A R reg x n ω ≤ W) :
    HistoryQuadraticWidthBound A R reg x n ω W := by
  exact ⟨h_nonneg, h_sum_le⟩

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- The packaged history-level quadratic-width input is monotone in the numeric bound. -/
lemma historyQuadraticWidthBound_mono {W W' : ℝ}
    (h_bound : HistoryQuadraticWidthBound A R reg x n ω W) (hW : W ≤ W') :
    HistoryQuadraticWidthBound A R reg x n ω W' := by
  exact ⟨h_bound.1, h_bound.2.trans hW⟩

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost surely, build the packaged history-level quadratic-width input from its two component
facts. -/
lemma historyQuadraticWidthBound_ae_of_nonneg_and_sum_ae_le {W : ℝ}
    (h_nonneg : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω))
    (h_sum_le : ∀ᵐ ω ∂P, historyQuadraticWidthSum A R reg x n ω ≤ W) :
    ∀ᵐ ω ∂P, HistoryQuadraticWidthBound A R reg x n ω W := by
  filter_upwards [h_nonneg, h_sum_le] with ω h_nonnegω h_sum_leω
  exact historyQuadraticWidthBound_of_nonneg_and_sum_le (A := A) (R := R)
    (reg := reg) (x := x) (n := n) (ω := ω) h_nonnegω h_sum_leω

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost surely, the packaged history-level quadratic-width input is monotone in the numeric
bound. -/
lemma historyQuadraticWidthBound_ae_mono {W W' : ℝ}
    (h_bound : ∀ᵐ ω ∂P, HistoryQuadraticWidthBound A R reg x n ω W) (hW : W ≤ W') :
    ∀ᵐ ω ∂P, HistoryQuadraticWidthBound A R reg x n ω W' := by
  filter_upwards [h_bound] with ω h_boundω
  exact historyQuadraticWidthBound_mono (A := A) (R := R) (reg := reg) (x := x)
    (n := n) (ω := ω) h_boundω hW

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- A capped quadratic-width sum bound gives the packaged history-level input whenever every
positive-time quadratic width form is nonnegative and at most `1`. -/
lemma historyQuadraticWidthBound_of_capped_sum_le {W : ℝ}
    (h_nonneg : ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω))
    (h_le_one : ∀ t, t ∈ range n → t ≠ 0 →
      widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω) ≤ 1)
    (h_capped_le : historyCappedQuadraticWidthSum A R reg x n ω ≤ W) :
    HistoryQuadraticWidthBound A R reg x n ω W := by
  refine historyQuadraticWidthBound_of_nonneg_and_sum_le (A := A) (R := R)
    (reg := reg) (x := x) (n := n) (ω := ω) h_nonneg ?_
  rw [historyQuadraticWidthSum_eq_historyCappedQuadraticWidthSum (A := A) (R := R)
    (reg := reg) (x := x) (n := n) (ω := ω) h_le_one]
  exact h_capped_le

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost surely, a capped quadratic-width sum bound gives the packaged history-level input
whenever every positive-time quadratic width form is almost surely nonnegative and at most `1`. -/
lemma historyQuadraticWidthBound_ae_of_capped_sum_ae_le {W : ℝ}
    (h_nonneg : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω))
    (h_le_one : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω) ≤ 1)
    (h_capped_le : ∀ᵐ ω ∂P, historyCappedQuadraticWidthSum A R reg x n ω ≤ W) :
    ∀ᵐ ω ∂P, HistoryQuadraticWidthBound A R reg x n ω W := by
  filter_upwards [h_nonneg, h_le_one, h_capped_le] with
    ω h_nonnegω h_le_oneω h_capped_leω
  exact historyQuadraticWidthBound_of_capped_sum_le (A := A) (R := R) (reg := reg)
    (x := x) (n := n) (ω := ω) h_nonnegω h_le_oneω h_capped_leω

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- The packaged history-level quadratic-width input implies the `widthSqSum` bound consumed by the
regret chain. -/
lemma widthSqSum_le_of_history_quadratic_width_bound {W : ℝ}
    (h_bound : HistoryQuadraticWidthBound A R reg x n ω W) :
    widthSqSum A reg x n ω ≤ W := by
  exact widthSqSum_le_of_history_quadratic_width_sum_le (A := A) (R := R) (reg := reg)
    (x := x) (n := n) (ω := ω) h_bound.1 h_bound.2

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost surely, the packaged history-level quadratic-width input implies the `widthSqSum` bound
consumed by the regret chain. -/
lemma widthSqSum_ae_le_of_history_quadratic_width_bound_ae {W : ℝ}
    (h_bound : ∀ᵐ ω ∂P, HistoryQuadraticWidthBound A R reg x n ω W) :
    ∀ᵐ ω ∂P, widthSqSum A reg x n ω ≤ W := by
  filter_upwards [h_bound] with ω h_boundω
  exact widthSqSum_le_of_history_quadratic_width_bound (A := A) (R := R) (reg := reg)
    (x := x) (n := n) (ω := ω) (W := W) h_boundω

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- A capped history-level quadratic-width sum bound implies the `widthSqSum` bound consumed by
the regret chain, provided the positive-time quadratic width forms are nonnegative and at most
`1`. -/
lemma widthSqSum_le_of_capped_history_quadratic_width_sum_le {W : ℝ}
    (h_nonneg : ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω))
    (h_le_one : ∀ t, t ∈ range n → t ≠ 0 →
      widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω) ≤ 1)
    (h_capped_le : historyCappedQuadraticWidthSum A R reg x n ω ≤ W) :
    widthSqSum A reg x n ω ≤ W := by
  exact widthSqSum_le_of_history_quadratic_width_bound (A := A) (R := R) (reg := reg)
    (x := x) (n := n) (ω := ω) (W := W)
    (historyQuadraticWidthBound_of_capped_sum_le (A := A) (R := R) (reg := reg)
      (x := x) (n := n) (ω := ω) h_nonneg h_le_one h_capped_le)

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Almost surely, a capped history-level quadratic-width sum bound implies the `widthSqSum` bound
consumed by the regret chain, provided the positive-time quadratic width forms are almost surely
nonnegative and at most `1`. -/
lemma widthSqSum_ae_le_of_capped_history_quadratic_width_sum_ae_le {W : ℝ}
    (h_nonneg : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      0 ≤ widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω))
    (h_le_one : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      widthQuadraticForm' reg x (t - 1) (IsAlgEnvSeq.hist A R (t - 1) ω) (A t ω) ≤ 1)
    (h_capped_le : ∀ᵐ ω ∂P, historyCappedQuadraticWidthSum A R reg x n ω ≤ W) :
    ∀ᵐ ω ∂P, widthSqSum A reg x n ω ≤ W := by
  exact widthSqSum_ae_le_of_history_quadratic_width_bound_ae (A := A) (R := R)
    (reg := reg) (x := x) (n := n) (P := P) (W := W)
    (historyQuadraticWidthBound_ae_of_capped_sum_ae_le (A := A) (R := R)
      (reg := reg) (x := x) (n := n) (P := P) (W := W) h_nonneg h_le_one
      h_capped_le)

lemma index_eq_index' (reg : ℝ) (β : ℕ → ℝ) (x : Fin K → Feature d)
    (a : Fin K) (n : ℕ) (ω : Ω) (hn : n ≠ 0) :
    index A R reg β x a n ω =
      index' reg β x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) a := by
  have htime : n + 1 = n - 1 + 2 := by grind
  simp [index, index', estimatedReward_eq_estimatedReward' (A := A) (R := R) reg x a n ω hn,
    width_eq_width' (A := A) (R := R) reg x a n ω hn, htime]

/-- The action at time `n + 1` is the finite-action LinUCB argmax for the observed history. -/
lemma arm_ae_eq_linUCBNextArm [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (n : ℕ) :
    A (n + 1) =ᵐ[P]
      fun ω ↦ nextArm hK reg β x n (IsAlgEnvSeq.hist A R n ω) := by
  have : Nonempty (Fin K) := Fin.pos_iff_nonempty.mp hK
  exact h.action_detAlgorithm_ae_eq n

/-- Almost surely, every positive-time action is the finite-action LinUCB argmax. -/
lemma arm_ae_all_eq [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P) :
    ∀ᵐ ω ∂P,
      ∀ n, A (n + 1) ω =
        nextArm hK reg β x n (IsAlgEnvSeq.hist A R n ω) := by
  simp_rw [ae_all_iff]
  exact fun n ↦ arm_ae_eq_linUCBNextArm h n

/-- Finite-action LinUCB chooses an arm maximizing the LinUCB index. -/
lemma index_le_index_arm [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (a : Fin K) (hn : n ≠ 0) :
    ∀ᵐ ω ∂P, index A R reg β x a n ω ≤ index A R reg β x (A n ω) n ω := by
  filter_upwards [arm_ae_eq_linUCBNextArm h (n - 1)] with ω h_arm
  have hn_succ : n - 1 + 1 = n := by grind
  simp only [hn_succ] at h_arm
  rw [index_eq_index' (A := A) (R := R) reg β x a n ω hn,
    index_eq_index' (A := A) (R := R) reg β x (A n ω) n ω hn]
  rw [h_arm]
  have : Nonempty (Fin K) := Fin.pos_iff_nonempty.mp hK
  exact isMaxOn_measurableArgmax (fun h a ↦ index' reg β x (n - 1) h a)
    (IsAlgEnvSeq.hist A R (n - 1) ω) a

/-- Almost surely, the selected arm maximizes the LinUCB index at every positive time. -/
lemma forall_index_le_index_arm [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (a : Fin K) :
    ∀ᵐ ω ∂P, ∀ n, n ≠ 0 →
      index A R reg β x a n ω ≤ index A R reg β x (A n ω) n ω := by
  simp_rw [ae_all_iff]
  exact fun n hn ↦ index_le_index_arm h a hn

/-- UCB-style package of the pathwise algorithm facts for finite-action LinUCB.

The process starts at the deterministic default arm, every positive-time action is the history-level
LinUCB argmax, and therefore the selected positive-time arm maximizes the process-level index. -/
lemma forall_arm_prop [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P) :
    ∀ᵐ ω ∂P,
      A 0 ω = ⟨0, hK⟩ ∧
      (∀ n, A (n + 1) ω =
        nextArm hK reg β x n (IsAlgEnvSeq.hist A R n ω)) ∧
      (∀ n, n ≠ 0 → ∀ a,
        index A R reg β x a n ω ≤ index A R reg β x (A n ω) n ω) := by
  have h_index_all :
      ∀ᵐ ω ∂P, ∀ a, ∀ n, n ≠ 0 →
        index A R reg β x a n ω ≤ index A R reg β x (A n ω) n ω := by
    simp_rw [ae_all_iff]
    intro a n hn
    exact index_le_index_arm h a hn
  filter_upwards [arm_zero (A := A) (R := R) (reg := reg) (β := β) (x := x)
    (ν := ν) h, arm_ae_all_eq (A := A) (R := R) (reg := reg) (β := β)
    (x := x) (ν := ν) h, h_index_all] with ω h0 h_arm h_indexω
  exact ⟨h0, h_arm, fun n hn a ↦ h_indexω a n hn⟩

end AlgorithmBehavior

end LinUCB

end Bandits
