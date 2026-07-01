/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: OpenAI, Fawad Haider
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.LinUCB.TextbookConfidenceBridge

/-!
# LinUCB Probability Bridges

Probability monotonicity, Ville/Markov transfers, and high-probability bridges
used by the LinUCB regret theorem.
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
/-- Probability monotonicity for almost-sure event inclusion. This keeps the LinUCB
high-probability wrappers focused on the mathematical event implication rather than repeating
measure boilerplate. -/
lemma probReal_event_le_of_ae_imp {E F : Ω → Prop}
    (h_imp : ∀ᵐ ω ∂P, E ω → F ω) :
    P.real {ω | E ω} ≤ P.real {ω | F ω} := by
  simp_rw [measureReal_def]
  gcongr 1
  · simp
  exact measure_mono_ae h_imp

/-- Failure-probability monotonicity for almost-sure event inclusion. -/
lemma probReal_failure_le_of_ae_imp {E F : Ω → Prop}
    (h_imp : ∀ᵐ ω ∂P, E ω → F ω) :
    P.real {ω | ¬ F ω} ≤ P.real {ω | ¬ E ω} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  filter_upwards [h_imp] with ω hω hF hE
  exact hF (hω hE)

omit [IsMarkovKernel ν] in
/-- If an event holds for every sample point, its failure probability is at most any nonnegative
budget. -/
lemma probReal_failure_le_of_forall {F : Ω → Prop} {δ : ℝ}
    (hF : ∀ ω, F ω) (hδ : 0 ≤ δ) :
    P.real {ω | ¬ F ω} ≤ δ := by
  have hfailure :
      P.real {ω | ¬ F ω} ≤ P.real {ω | ¬ True} := by
    refine probReal_failure_le_of_ae_imp (P := P) (E := fun _ ↦ True) (F := F) ?_
    exact Filter.Eventually.of_forall fun ω _ ↦ hF ω
  exact hfailure.trans (by simpa [measureReal_def] using hδ)

lemma probReal_union_le (E F : Set Ω) :
    P.real (E ∪ F) ≤ P.real E + P.real F := by
  rw [measureReal_def, measureReal_def, measureReal_def]
  calc
    (P (E ∪ F)).toReal ≤ (P E + P F).toReal := by
      exact ENNReal.toReal_mono (by finiteness) (measure_union_le E F)
    _ = (P E).toReal + (P F).toReal := by
      exact ENNReal.toReal_add (by finiteness) (by finiteness)

omit [IsMarkovKernel ν] in
/-- Convert a real-valued failure-probability bound into a high-probability success bound.

No measurability assumption is needed: the proof only uses the union bound for
`E ∪ Eᶜ = univ`, which is valid for the outer-measure value used by `P.real`. -/
lemma probReal_event_ge_of_failure_le {E : Ω → Prop} {δ : ℝ}
    (h_failure : P.real {ω | ¬ E ω} ≤ δ) :
    1 - δ ≤ P.real {ω | E ω} := by
  have h_union :
      1 ≤ P.real {ω | E ω} + P.real {ω | ¬ E ω} := by
    calc
      1 = P.real Set.univ := by simp [measureReal_def]
      _ = P.real ({ω | E ω} ∪ {ω | ¬ E ω}) := by
            congr 1
            ext ω
            by_cases hω : E ω <;> simp [hω]
      _ ≤ P.real {ω | E ω} + P.real {ω | ¬ E ω} :=
            probReal_union_le (P := P) {ω | E ω} {ω | ¬ E ω}
  linarith

omit [IsMarkovKernel ν] in
lemma probReal_textbookMixtureUpTo_le_textbookNoiseUpTo
    {σ2 : ℝ≥0} {δ : ℝ}
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg) :
    P.real {ω | LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤
      P.real {ω | LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_mixω ↦
    LinUCBTextbookSelfNormalizedNoiseEventUpTo.of_mixtureBound_of_reg_pos
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) (ω := ω)
      hσ2_pos hδ_pos hreg_pos h_mixω

omit [IsMarkovKernel ν] in
/-- High-probability transfer from the Gaussian-mixture event to the textbook self-normalized
centered-noise event. -/
lemma probReal_textbookSelfNormalizedNoiseEventUpTo_ge_of_mixtureUpTo_ge
    {σ2 : ℝ≥0} {δ : ℝ}
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    (h_mix_prob :
      1 - δ ≤
        P.real {ω | LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω}) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} :=
  h_mix_prob.trans
    (probReal_textbookMixtureUpTo_le_textbookNoiseUpTo (A := A) (R := R)
      (reg := reg) (x := x) (ν := ν) (n := n) (P := P)
      hσ2_pos hδ_pos hreg_pos)

omit [IsMarkovKernel ν] in
/-- Failure-probability transfer from the Gaussian-mixture event to the textbook self-normalized
centered-noise event. -/
lemma probReal_textbookSelfNormalizedNoiseEventUpTo_failure_le_of_mixtureUpTo_failure_le
    {σ2 : ℝ≥0} {δ : ℝ}
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    (h_mix_failure :
      P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤ δ) :
    P.real {ω | ¬ LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} ≤ δ := by
  refine le_trans ?_ h_mix_failure
  refine probReal_failure_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_mixω ↦
    LinUCBTextbookSelfNormalizedNoiseEventUpTo.of_mixtureBound_of_reg_pos
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) (ω := ω)
      hσ2_pos hδ_pos hreg_pos h_mixω

omit [IsMarkovKernel ν] in
lemma probReal_textbookMixtureUpTo_failure_le_boundedStoppedMixture_ge_inv_delta
    {σ2 : ℝ≥0} {δ : ℝ} :
    P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤
      P.real {ω | 1 / δ ≤
        boundedStoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω hfail ↦
    le_of_lt
      (inv_delta_lt_boundedStoppedTextbookMixtureStatistic_of_mixture_failure
        (K := K) (d := d) (A := A) (R := R) (reg := reg) (σ2 := σ2)
        (δ := δ) (x := x) (ν := ν) (n := n) (ω := ω) hfail)

omit [IsMarkovKernel ν] in
/-- Markov/Ville-style probability step for the bounded stopped textbook Gaussian-mixture
statistic.

If the bounded stopped mixture statistic has expectation at most one, then the horizon-local
mixture event fails with probability at most `δ`. -/
lemma probReal_textbookMixtureUpTo_failure_le_of_boundedStoppedMixture_integral_le
    {σ2 : ℝ≥0} {δ : ℝ}
    (hδ_pos : 0 < δ)
    (hstop_integrable :
      Integrable
        (fun ω ↦ boundedStoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω) P)
    (hstop_integral :
      (∫ ω, boundedStoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω ∂P) ≤ 1) :
    P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤ δ := by
  have hthreshold_pos : 0 < 1 / δ := one_div_pos.mpr hδ_pos
  have hstop_nonneg :
      0 ≤ᵐ[P] fun ω ↦ boundedStoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω :=
    Filter.Eventually.of_forall fun ω ↦
      boundedStoppedTextbookMixtureStatistic_nonneg (A := A) (R := R) (reg := reg)
        (σ2 := σ2) (δ := δ) (x := x) (ν := ν) (n := n) (ω := ω)
  have hmarkov :
      (1 / δ) *
          P.real {ω | 1 / δ ≤
            boundedStoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω} ≤
        ∫ ω, boundedStoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω ∂P :=
    mul_meas_ge_le_integral_of_nonneg
      (μ := P)
      (f := fun ω ↦ boundedStoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω)
      hstop_nonneg hstop_integrable (1 / δ)
  have hfailure_subset :=
    probReal_textbookMixtureUpTo_failure_le_boundedStoppedMixture_ge_inv_delta
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
      (n := n) (P := P)
  have hfailure_mul :
      (1 / δ) *
          P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤
        1 := by
    calc
      (1 / δ) *
          P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω}
          ≤ (1 / δ) *
              P.real {ω | 1 / δ ≤
                boundedStoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω} := by
            exact mul_le_mul_of_nonneg_left hfailure_subset hthreshold_pos.le
      _ ≤ ∫ ω, boundedStoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω ∂P :=
            hmarkov
      _ ≤ 1 := hstop_integral
  have hdiv :
      P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} / δ ≤
        1 := by
    simpa [one_div, div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hfailure_mul
  have hle :
      P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤
        1 * δ :=
    (div_le_iff₀ hδ_pos).mp hdiv
  simpa using hle

omit [IsMarkovKernel ν] in
lemma probReal_textbookMixtureUpTo_failure_le_of_boundedStoppedMixture_supermartingale
    {σ2 : ℝ≥0} {δ : ℝ}
    (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    {ℱ : Filtration ℕ mΩ} [SigmaFiniteFiltration P ℱ]
    (hM :
      Supermartingale
        (fun t ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν t ω) ℱ P) :
    P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤ δ := by
  exact probReal_textbookMixtureUpTo_failure_le_of_boundedStoppedMixture_integral_le
    (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
    (n := n) (P := P) hδ_pos
    (integrable_boundedStoppedTextbookMixtureStatistic_of_supermartingale
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
      (n := n) (P := P) (ℱ := ℱ) hM)
    (integral_boundedStoppedTextbookMixtureStatistic_le_one_of_supermartingale
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
      (n := n) (P := P) (ℱ := ℱ) hM hreg_pos)

lemma probReal_textbookMixtureUpTo_failure_le_of_textbookGaussianMixtureInput
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    (μlambda : Measure (Feature d)) [SFinite μlambda]
    (h_mix : TextbookGaussianMixtureInput A R ν reg σ2 x P μlambda) :
    P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤ δ := by
  let ℱ := IsAlgEnvSeq.filtrationAction h.measurable_action h.measurable_feedback
  have hM :
      Supermartingale
        (fun t ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν t ω) ℱ P :=
    supermartingale_textbookMixtureStatistic_of_directionalMixture_global_prod_integrable
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
      h hν μlambda h_mix
  exact probReal_textbookMixtureUpTo_failure_le_of_boundedStoppedMixture_supermartingale
    (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
    (n := n) (P := P) hδ_pos hreg_pos (ℱ := ℱ) hM

lemma probReal_textbookMixtureUpTo_failure_le_of_textbookGaussianPriorInput
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    (h_prior : TextbookGaussianPriorInput A R ν reg σ2 x P) :
    P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤ δ := by
  let ℱ := IsAlgEnvSeq.filtrationAction h.measurable_action h.measurable_feedback
  have hM :
      Supermartingale
        (fun t ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν t ω) ℱ P :=
    supermartingale_textbookMixtureStatistic_of_textbookGaussianPriorInput
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
      h hν h_prior
  exact probReal_textbookMixtureUpTo_failure_le_of_boundedStoppedMixture_supermartingale
    (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
    (n := n) (P := P) hδ_pos hreg_pos (ℱ := ℱ) hM

/-- High-probability textbook mixture bound from the concrete Gaussian direction prior used in the
textbook method-of-mixtures proof. -/
lemma probReal_textbookMixtureUpTo_ge_of_textbookGaussianPriorInput
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    (h_prior : TextbookGaussianPriorInput A R ν reg σ2 x P) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} :=
  probReal_event_ge_of_failure_le (P := P)
    (probReal_textbookMixtureUpTo_failure_le_of_textbookGaussianPriorInput
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
      (n := n) (P := P) h hν hδ_pos hreg_pos h_prior)

/-- Failure-probability self-normalized noise bound from a textbook Gaussian-mixture input. -/
lemma probReal_textbookNoise_failure_le_of_textbookGaussianMixtureInput
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    (μlambda : Measure (Feature d)) [SFinite μlambda]
    (h_mix : TextbookGaussianMixtureInput A R ν reg σ2 x P μlambda) :
    P.real {ω | ¬ LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} ≤
      δ :=
  probReal_textbookSelfNormalizedNoiseEventUpTo_failure_le_of_mixtureUpTo_failure_le
    (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
    (n := n) (P := P) hσ2_pos hδ_pos hreg_pos
    (probReal_textbookMixtureUpTo_failure_le_of_textbookGaussianMixtureInput
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
      (n := n) (P := P) h hν hδ_pos hreg_pos μlambda h_mix)

lemma probReal_textbookNoise_failure_le_of_textbookGaussianPriorInput
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    (h_prior : TextbookGaussianPriorInput A R ν reg σ2 x P) :
    P.real {ω | ¬ LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} ≤
      δ := by
  have h_mix : TextbookGaussianMixtureInput A R ν reg σ2 x P
      (gaussianDirectionMeasure d reg σ2) :=
    TextbookGaussianPriorInput.toMixtureInput (A := A) (R := R) (reg := reg)
      (σ2 := σ2) (x := x) (ν := ν) (P := P) h_prior
  exact probReal_textbookNoise_failure_le_of_textbookGaussianMixtureInput
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
    (n := n) (P := P) h hν hσ2_pos hδ_pos hreg_pos
    (gaussianDirectionMeasure d reg σ2) h_mix

/-- High-probability self-normalized noise bound from the concrete Gaussian direction prior used in
the textbook method-of-mixtures proof. -/
lemma probReal_textbookNoise_ge_of_textbookGaussianPriorInput
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    (h_prior : TextbookGaussianPriorInput A R ν reg σ2 x P) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} :=
  probReal_textbookSelfNormalizedNoiseEventUpTo_ge_of_mixtureUpTo_ge
    (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
    (n := n) (P := P) hσ2_pos hδ_pos hreg_pos
    (probReal_textbookMixtureUpTo_ge_of_textbookGaussianPriorInput
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
      (n := n) (P := P) h hν hδ_pos hreg_pos h_prior)

omit [IsMarkovKernel ν] in
lemma probReal_textbookNoiseUpTo_le_centeredNoiseBiasUpTo
    (θ : Feature d) (S2 : ℝ) {σ2 : ℝ≥0} {δ : ℝ}
    (hreg_pos : 0 < reg)
    (hθ : ParameterSqNormBound θ S2)
    (h_budget : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      (√(textbookSelfNormalizedNoiseBound σ2 δ (designDetRatio A reg x t ω)) +
        √(reg * S2)) ^ 2 ≤ β (t + 1)) :
    P.real {ω | LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} ≤
      P.real {ω | LinUCBCenteredNoiseBiasConfidenceEventUpTo A R reg β x ν θ n ω} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  filter_upwards [h_budget] with ω h_budgetω h_noiseω
  exact LinUCBCenteredNoiseBiasConfidenceEventUpTo.of_textbookSelfNormalizedNoise
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
    (ω := ω) θ S2 hreg_pos hθ h_noiseω h_budgetω

omit [IsMarkovKernel ν] in
/-- High-probability transfer from the textbook determinant-ratio self-normalized event to the
centered-noise-plus-bias event. -/
lemma probReal_centeredNoiseBiasUpTo_ge_of_textbookNoiseUpTo_ge
    (θ : Feature d) (S2 : ℝ) {σ2 : ℝ≥0} {δ : ℝ}
    (hreg_pos : 0 < reg)
    (hθ : ParameterSqNormBound θ S2)
    (h_budget : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      (√(textbookSelfNormalizedNoiseBound σ2 δ (designDetRatio A reg x t ω)) +
        √(reg * S2)) ^ 2 ≤ β (t + 1))
    (h_noise_prob :
      1 - δ ≤
        P.real {ω |
          LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω}) :
    1 - δ ≤
      P.real {ω | LinUCBCenteredNoiseBiasConfidenceEventUpTo A R reg β x ν θ n ω} :=
  h_noise_prob.trans
    (probReal_textbookNoiseUpTo_le_centeredNoiseBiasUpTo (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (P := P)
      θ S2 hreg_pos hθ h_budget)

end AlgorithmBehavior

end LinUCB

end Bandits
