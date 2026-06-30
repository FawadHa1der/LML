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

/-- Fixed-direction exponent tail bound in the log-threshold form used just before the textbook
Gaussian-mixture argument.

The previous fixed-direction lemma bounds the probability of `exp(exponent) ≥ 1 / δ`. Since
`δ > 0`, this is equivalent to `exponent ≥ log (1 / δ)`. -/
lemma probReal_centeredResponse_directionalExponent_ge_log_inv_delta_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (v : Feature d) (u : ℝ) {δ : ℝ} (hδ_pos : 0 < δ) :
    P.real {ω |
      Real.log (1 / δ) ≤
        u * dotProduct v (centeredResponseVector A R ν x n ω) -
          ((σ2 : ℝ) *
            (dotProduct v (Matrix.mulVec (designMatrix A reg x n ω) v) -
              reg * dotProduct v v)) *
              u ^ 2 / 2} ≤ δ := by
  have hmono :
      P.real {ω |
        Real.log (1 / δ) ≤
          u * dotProduct v (centeredResponseVector A R ν x n ω) -
            ((σ2 : ℝ) *
              (dotProduct v (Matrix.mulVec (designMatrix A reg x n ω) v) -
                reg * dotProduct v v)) *
                u ^ 2 / 2} ≤
        P.real {ω |
          1 / δ ≤
            Real.exp
              (u * dotProduct v (centeredResponseVector A R ν x n ω) -
                ((σ2 : ℝ) *
                  (dotProduct v (Matrix.mulVec (designMatrix A reg x n ω) v) -
                    reg * dotProduct v v)) *
                    u ^ 2 / 2)} := by
    refine probReal_event_le_of_ae_imp (P := P) ?_
    exact Filter.Eventually.of_forall fun ω hω ↦ by
      exact (Real.log_le_iff_le_exp (one_div_pos.mpr hδ_pos)).mp hω
  exact hmono.trans
    (probReal_exp_centeredResponse_sub_designMatrix_minus_reg_norm_ge_inv_delta_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      h hν v u hδ_pos)

/-- Fixed-vector exponential-process tail bound in the named form used by the textbook
Gaussian-mixture proof. -/
lemma probReal_exp_centeredResponseDirectionalExponent_ge_inv_delta_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (lambda : Feature d) {δ : ℝ} (hδ_pos : 0 < δ) :
    P.real {ω |
      1 / δ ≤
        Real.exp (centeredResponseDirectionalExponent A R ν reg σ2 x n ω lambda)} ≤ δ := by
  simpa [centeredResponseDirectionalExponent, mul_assoc] using
    probReal_exp_centeredResponse_sub_designMatrix_minus_reg_norm_ge_inv_delta_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      h hν lambda (1 : ℝ) hδ_pos

/-- Fixed-vector log-threshold tail bound in the form used by the textbook Gaussian-mixture proof.

This is the canonical `λ`-direction version of
`probReal_centeredResponse_directionalExponent_ge_log_inv_delta_le`, obtained by taking `u = 1`. -/
lemma probReal_centeredResponseDirectionalExponent_ge_log_inv_delta_le
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (lambda : Feature d) {δ : ℝ} (hδ_pos : 0 < δ) :
    P.real {ω |
      Real.log (1 / δ) ≤
        centeredResponseDirectionalExponent A R ν reg σ2 x n ω lambda} ≤ δ := by
  have hmono :
      P.real {ω |
        Real.log (1 / δ) ≤
          centeredResponseDirectionalExponent A R ν reg σ2 x n ω lambda} ≤
        P.real {ω |
          1 / δ ≤
            Real.exp
              (centeredResponseDirectionalExponent A R ν reg σ2 x n ω lambda)} := by
    refine probReal_event_le_of_ae_imp (P := P) ?_
    exact Filter.Eventually.of_forall fun ω hω ↦ by
      exact (Real.log_le_iff_le_exp (one_div_pos.mpr hδ_pos)).mp hω
  exact hmono.trans
    (probReal_exp_centeredResponseDirectionalExponent_ge_inv_delta_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      h hν lambda hδ_pos)

omit [IsMarkovKernel ν] in
/-- Failure-probability monotonicity for almost-sure event inclusion. If `E` implies `F` almost
surely, then failure of `F` is contained in failure of `E` almost surely. -/
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

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- Every real-valued probability is nonnegative. -/
lemma probReal_nonneg (E : Set Ω) :
    0 ≤ P.real E := by
  simp [measureReal_def]

omit [IsMarkovKernel ν] in
/-- Every real-valued probability is at most one under a probability measure. -/
lemma probReal_le_one (E : Set Ω) :
    P.real E ≤ 1 := by
  rw [measureReal_def]
  calc
    (P E).toReal ≤ (P Set.univ).toReal := by
      exact ENNReal.toReal_mono (by finiteness) (measure_mono (Set.subset_univ E))
    _ = 1 := by simp

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- If `δ ≥ 1`, any event automatically has probability at least `1 - δ`. -/
lemma probReal_event_ge_of_one_le_delta {E : Ω → Prop} {δ : ℝ}
    (hδ : 1 ≤ δ) :
    1 - δ ≤ P.real {ω | E ω} := by
  exact (by linarith : 1 - δ ≤ 0).trans (probReal_nonneg (P := P) {ω | E ω})

omit [IsMarkovKernel ν] in
/-- If `δ ≥ 1`, every event has failure probability at most `δ`. -/
lemma probReal_failure_le_of_one_le_delta {E : Ω → Prop} {δ : ℝ}
    (hδ : 1 ≤ δ) :
    P.real {ω | ¬ E ω} ≤ δ :=
  (probReal_le_one (P := P) {ω | ¬ E ω}).trans hδ

omit [IsMarkovKernel ν] in
/-- Real-valued finite union bound for two events. -/
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
/-- Real-valued finite union bound over a finite index set. -/
lemma probReal_biUnion_finset_le_sum {ι : Type*} (I : Finset ι) (E : ι → Set Ω) :
    P.real (⋃ i ∈ I, E i) ≤ ∑ i ∈ I, P.real (E i) := by
  rw [measureReal_def]
  calc
    (P (⋃ i ∈ I, E i)).toReal ≤ (∑ i ∈ I, P (E i)).toReal := by
      exact ENNReal.toReal_mono
        (ENNReal.sum_ne_top.2 fun i _hi ↦ measure_ne_top P (E i))
        (measure_biUnion_finset_le I E)
    _ = ∑ i ∈ I, P.real (E i) := by
      simp_rw [measureReal_def]
      exact ENNReal.toReal_sum fun i _hi ↦ measure_ne_top P (E i)

omit [IsMarkovKernel ν] in
/-- Real-valued finite union bound over a finite type. -/
lemma probReal_iUnion_fintype_le_sum {ι : Type*} [Fintype ι] (E : ι → Set Ω) :
    P.real (⋃ i, E i) ≤ ∑ i, P.real (E i) := by
  simpa using probReal_biUnion_finset_le_sum (P := P) Finset.univ E

omit [IsMarkovKernel ν] in
/-- Probability monotonicity for the finite-horizon textbook confidence decomposition. If the
self-normalized centered-noise event holds, and its radius plus the deterministic ridge-bias radius
fits under `β`, then the centered-noise-plus-bias event holds. -/
lemma probReal_centeredNoiseConfidenceEventUpTo_le_centeredNoiseBiasConfidenceEventUpTo
    (θ : Feature d) (S2 : ℝ)
    (hreg_pos : 0 < reg)
    (hθ : ParameterSqNormBound θ S2)
    {noiseBudget : ℕ → ℝ}
    (h_budget : ∀ t, t ∈ range n → t ≠ 0 →
      (√(noiseBudget (t + 1)) + √(reg * S2)) ^ 2 ≤ β (t + 1)) :
    P.real {ω | LinUCBCenteredNoiseConfidenceEventUpTo A R reg noiseBudget x ν n ω} ≤
      P.real {ω | LinUCBCenteredNoiseBiasConfidenceEventUpTo A R reg β x ν θ n ω} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_noiseω ↦
    LinUCBCenteredNoiseBiasConfidenceEventUpTo.of_centeredNoise (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (ω := ω)
      θ S2 hreg_pos hθ h_noiseω h_budget

omit [IsMarkovKernel ν] in
/-- High-probability transfer from a finite-horizon centered-noise event to the existing
centered-noise-plus-bias event. -/
lemma probReal_centeredNoiseBiasConfidenceEventUpTo_ge_of_centeredNoiseConfidenceEventUpTo_ge
    (θ : Feature d) (S2 : ℝ)
    (hreg_pos : 0 < reg)
    (hθ : ParameterSqNormBound θ S2)
    {noiseBudget : ℕ → ℝ} {δ : ℝ}
    (h_budget : ∀ t, t ∈ range n → t ≠ 0 →
      (√(noiseBudget (t + 1)) + √(reg * S2)) ^ 2 ≤ β (t + 1))
    (h_noise_prob :
      1 - δ ≤ P.real {ω |
        LinUCBCenteredNoiseConfidenceEventUpTo A R reg noiseBudget x ν n ω}) :
    1 - δ ≤
      P.real {ω | LinUCBCenteredNoiseBiasConfidenceEventUpTo A R reg β x ν θ n ω} :=
  h_noise_prob.trans
    (probReal_centeredNoiseConfidenceEventUpTo_le_centeredNoiseBiasConfidenceEventUpTo
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      (P := P) θ S2 hreg_pos hθ h_budget)

omit [IsMarkovKernel ν] in
/-- Failure-probability transfer from a finite-horizon centered-noise event to the existing
centered-noise-plus-bias event. -/
lemma probReal_centeredNoiseBiasUpTo_failure_le_of_centeredNoiseUpTo_failure_le
    (θ : Feature d) (S2 : ℝ)
    (hreg_pos : 0 < reg)
    (hθ : ParameterSqNormBound θ S2)
    {noiseBudget : ℕ → ℝ} {δ : ℝ}
    (h_budget : ∀ t, t ∈ range n → t ≠ 0 →
      (√(noiseBudget (t + 1)) + √(reg * S2)) ^ 2 ≤ β (t + 1))
    (h_noise_failure :
      P.real {ω | ¬ LinUCBCenteredNoiseConfidenceEventUpTo A R reg noiseBudget x ν n ω} ≤ δ) :
    P.real {ω | ¬ LinUCBCenteredNoiseBiasConfidenceEventUpTo A R reg β x ν θ n ω} ≤ δ := by
  refine le_trans ?_ h_noise_failure
  refine probReal_failure_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω h_noiseω ↦
    LinUCBCenteredNoiseBiasConfidenceEventUpTo.of_centeredNoise (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (ω := ω)
      θ S2 hreg_pos hθ h_noiseω h_budget

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- For `δ ≥ 1`, the textbook self-normalized noise event automatically satisfies the
high-probability lower bound `1 - δ`.

This isolates the trivial probability range, so the future Gaussian-mixture concentration theorem
only needs to prove the nontrivial case `δ < 1`. -/
lemma probReal_textbookSelfNormalizedNoiseEventUpTo_ge_of_one_le_delta
    {σ2 : ℝ≥0} {δ : ℝ} (hδ : 1 ≤ δ) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} :=
  probReal_event_ge_of_one_le_delta (P := P) hδ

omit [IsMarkovKernel ν] in
/-- For `δ ≥ 1`, the failure probability of the textbook self-normalized noise event is
automatically at most `δ`. -/
lemma probReal_textbookSelfNormalizedNoiseEventUpTo_failure_le_of_one_le_delta
    {σ2 : ℝ≥0} {δ : ℝ} (hδ : 1 ≤ δ) :
    P.real {ω | ¬ LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} ≤ δ :=
  probReal_failure_le_of_one_le_delta (P := P) hδ

omit [IsMarkovKernel ν] in
/-- Probability monotonicity from the textbook Gaussian-mixture event to the textbook
self-normalized centered-noise event. -/
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
/-- Failure of the textbook Gaussian-mixture event is contained in the event that the stopped
mixture statistic is at least `1 / δ`. -/
lemma probReal_textbookMixtureUpTo_failure_le_stoppedMixture_ge_inv_delta
    {σ2 : ℝ≥0} {δ : ℝ} :
    P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤
      P.real {ω | 1 / δ ≤ stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω} := by
  refine probReal_event_le_of_ae_imp (P := P) ?_
  exact Filter.Eventually.of_forall fun ω hfail ↦
    le_of_lt
      (inv_delta_lt_stoppedTextbookMixtureStatistic_of_mixture_failure
        (K := K) (d := d) (A := A) (R := R) (reg := reg) (σ2 := σ2)
        (δ := δ) (x := x) (ν := ν) (n := n) (ω := ω) hfail)

omit [IsMarkovKernel ν] in
/-- Markov/Ville-style probability step for the stopped textbook Gaussian-mixture statistic.

If the stopped mixture statistic has expectation at most one, then the horizon-local mixture event
fails with probability at most `δ`. The remaining textbook concentration work is exactly to prove
the two analytic inputs here: integrability and the stopped expectation bound, using the Gaussian
mixture/supermartingale argument. -/
lemma probReal_textbookMixtureUpTo_failure_le_of_stoppedMixture_integral_le
    {σ2 : ℝ≥0} {δ : ℝ}
    (hδ_pos : 0 < δ)
    (hstop_integrable :
      Integrable (fun ω ↦ stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω) P)
    (hstop_integral :
      (∫ ω, stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω ∂P) ≤ 1) :
    P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤ δ := by
  have hthreshold_pos : 0 < 1 / δ := one_div_pos.mpr hδ_pos
  have hstop_nonneg :
      0 ≤ᵐ[P] fun ω ↦ stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω :=
    Filter.Eventually.of_forall fun ω ↦
      stoppedTextbookMixtureStatistic_nonneg (A := A) (R := R) (reg := reg)
        (σ2 := σ2) (δ := δ) (x := x) (ν := ν) (n := n) (ω := ω)
  have hmarkov :
      (1 / δ) *
          P.real {ω | 1 / δ ≤ stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω} ≤
        ∫ ω, stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω ∂P :=
    mul_meas_ge_le_integral_of_nonneg
      (μ := P)
      (f := fun ω ↦ stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω)
      hstop_nonneg hstop_integrable (1 / δ)
  have hfailure_subset :=
    probReal_textbookMixtureUpTo_failure_le_stoppedMixture_ge_inv_delta
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
      (n := n) (P := P)
  have hfailure_mul :
      (1 / δ) *
          P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤ 1 := by
    calc
      (1 / δ) *
          P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω}
          ≤ (1 / δ) *
              P.real {ω | 1 / δ ≤
                stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω} := by
            exact mul_le_mul_of_nonneg_left hfailure_subset hthreshold_pos.le
      _ ≤ ∫ ω, stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω ∂P := hmarkov
      _ ≤ 1 := hstop_integral
  have hdiv :
      P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} / δ ≤ 1 := by
    simpa [one_div, div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hfailure_mul
  have hle :
      P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤
        1 * δ :=
    (div_le_iff₀ hδ_pos).mp hdiv
  simpa using hle

omit [IsMarkovKernel ν] in
/-- High-probability form of
`probReal_textbookMixtureUpTo_failure_le_of_stoppedMixture_integral_le`. -/
lemma probReal_textbookMixtureUpTo_ge_of_stoppedMixture_integral_le
    {σ2 : ℝ≥0} {δ : ℝ}
    (hδ_pos : 0 < δ)
    (hstop_integrable :
      Integrable (fun ω ↦ stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω) P)
    (hstop_integral :
      (∫ ω, stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω ∂P) ≤ 1) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} :=
  probReal_event_ge_of_failure_le (P := P)
    (probReal_textbookMixtureUpTo_failure_le_of_stoppedMixture_integral_le
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
      (n := n) (P := P) hδ_pos hstop_integrable hstop_integral)

omit [IsMarkovKernel ν] in
/-- Failure of the textbook Gaussian-mixture event is contained in the event that the
bounded-horizon stopped mixture statistic is at least `1 / δ`.

This is the same event inclusion as
`probReal_textbookMixtureUpTo_failure_le_stoppedMixture_ge_inv_delta`, but it uses
`boundedStoppedTextbookMixtureStatistic`, whose stopping time falls back to the deterministic
horizon `n` when there is no crossing. That bounded fallback is the form used by optional
stopping. -/
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
/-- High-probability form of
`probReal_textbookMixtureUpTo_failure_le_of_boundedStoppedMixture_integral_le`. -/
lemma probReal_textbookMixtureUpTo_ge_of_boundedStoppedMixture_integral_le
    {σ2 : ℝ≥0} {δ : ℝ}
    (hδ_pos : 0 < δ)
    (hstop_integrable :
      Integrable
        (fun ω ↦ boundedStoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω) P)
    (hstop_integral :
      (∫ ω, boundedStoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω ∂P) ≤ 1) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} :=
  probReal_event_ge_of_failure_le (P := P)
    (probReal_textbookMixtureUpTo_failure_le_of_boundedStoppedMixture_integral_le
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
      (n := n) (P := P) hδ_pos hstop_integrable hstop_integral)

omit [IsMarkovKernel ν] in
/-- Failure-probability textbook mixture bound from the supermartingale form of the
Gaussian-mixture argument.

Compared with the older stopped-statistic interface, this theorem has no separate stopping-time,
integrability, or stopped-expectation assumptions. Those are derived from the bounded first
crossing time and optional stopping. The remaining probabilistic input is exactly the process-level
supermartingale statement for the textbook mixture statistic. -/
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

omit [IsMarkovKernel ν] in
/-- High-probability textbook mixture bound from the supermartingale form of the
Gaussian-mixture argument. -/
lemma probReal_textbookMixtureUpTo_ge_of_boundedStoppedMixture_supermartingale
    {σ2 : ℝ≥0} {δ : ℝ}
    (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    {ℱ : Filtration ℕ mΩ} [SigmaFiniteFiltration P ℱ]
    (hM :
      Supermartingale
        (fun t ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν t ω) ℱ P) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} :=
  probReal_event_ge_of_failure_le (P := P)
    (probReal_textbookMixtureUpTo_failure_le_of_boundedStoppedMixture_supermartingale
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
      (n := n) (P := P) hδ_pos hreg_pos (ℱ := ℱ) hM)

/-- Failure-probability textbook mixture bound from the named Gaussian-mixture input.

This is the probability-layer concentration interface closest to the textbook proof. A future
multivariate Gaussian integral theorem should prove `TextbookGaussianMixtureInput` for the
textbook Gaussian direction measure; this lemma then turns that analytic input into the finite
horizon mixture event used by the regret theorem. -/
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

/-- High-probability textbook mixture bound from the named Gaussian-mixture input. -/
lemma probReal_textbookMixtureUpTo_ge_of_textbookGaussianMixtureInput
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    (μlambda : Measure (Feature d)) [SFinite μlambda]
    (h_mix : TextbookGaussianMixtureInput A R ν reg σ2 x P μlambda) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} :=
  probReal_event_ge_of_failure_le (P := P)
    (probReal_textbookMixtureUpTo_failure_le_of_textbookGaussianMixtureInput
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
      (n := n) (P := P) h hν hδ_pos hreg_pos μlambda h_mix)

/-- Failure-probability textbook mixture bound from the concrete Gaussian direction prior used in
the textbook method-of-mixtures proof. -/
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

/-- Failure-probability textbook mixture bound from the remaining unshifted anisotropic Gaussian
determinant integral.

This is the same probability theorem as
`probReal_textbookMixtureUpTo_failure_le_of_textbookGaussianPriorInput`, but exposes the concrete
analytic obligations left by the completed-square proof: the unshifted positive-definite Gaussian
integral and product-integrability for the Gaussian prior, both only at positive times. The
time-zero determinant-integral identity and product-integrability case are discharged internally. -/
lemma probReal_textbookMixtureUpTo_failure_le_of_anisotropicKernelIntegral
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hδ_pos : 0 < δ) (hreg_pos : 0 < reg) (hσ2_pos : 0 < (σ2 : ℝ))
    (h_kernel_pos : ∀ i, i ≠ 0 → ∀ᵐ ω ∂P,
      Real.rpow ((σ2 : ℝ) * reg / (2 * Real.pi)) ((d : ℝ) / 2) *
          (∫ lambda, anisotropicGaussianKernel A reg σ2 x i ω lambda) =
        1 / √(designDetRatio A reg x i ω))
    (h_prod_pos : ∀ i, i ≠ 0 →
      Integrable
        (Function.uncurry fun ω lambda ↦
          Real.exp (centeredResponseDirectionalExponent A R ν reg σ2 x i ω lambda))
        (P.prod (gaussianDirectionMeasure d reg σ2))) :
    P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤ δ :=
  probReal_textbookMixtureUpTo_failure_le_of_textbookGaussianPriorInput
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
    (n := n) (P := P) h hν hδ_pos hreg_pos
    (textbookGaussianPriorInput_of_anisotropicKernelIntegral_posTime
      (A := A) (R := R) (ν := ν) (reg := reg) (x := x) (P := P)
      hreg_pos hσ2_pos h_kernel_pos h_prod_pos)

/-- High-probability textbook mixture bound from the remaining unshifted anisotropic Gaussian
determinant integral. -/
lemma probReal_textbookMixtureUpTo_ge_of_anisotropicKernelIntegral
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hδ_pos : 0 < δ) (hreg_pos : 0 < reg) (hσ2_pos : 0 < (σ2 : ℝ))
    (h_kernel_pos : ∀ i, i ≠ 0 → ∀ᵐ ω ∂P,
      Real.rpow ((σ2 : ℝ) * reg / (2 * Real.pi)) ((d : ℝ) / 2) *
          (∫ lambda, anisotropicGaussianKernel A reg σ2 x i ω lambda) =
        1 / √(designDetRatio A reg x i ω))
    (h_prod_pos : ∀ i, i ≠ 0 →
      Integrable
        (Function.uncurry fun ω lambda ↦
          Real.exp (centeredResponseDirectionalExponent A R ν reg σ2 x i ω lambda))
        (P.prod (gaussianDirectionMeasure d reg σ2))) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} :=
  probReal_event_ge_of_failure_le (P := P)
    (probReal_textbookMixtureUpTo_failure_le_of_anisotropicKernelIntegral
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
      (n := n) (P := P) h hν hδ_pos hreg_pos hσ2_pos h_kernel_pos h_prod_pos)

/-- Failure-probability textbook mixture bound after discharging the anisotropic determinant
integral by spectral diagonalization of the LinUCB design matrix.

The determinant integral and product-integrability of the fixed-direction exponential process are
both discharged internally from the spectral integral theorem and the scalar exponential
supermartingale bound. -/
lemma probReal_textbookMixtureUpTo_failure_le_of_designEigenvectorIntegral
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hδ_pos : 0 < δ) (hreg_pos : 0 < reg) (hσ2_pos : 0 < (σ2 : ℝ)) :
    P.real {ω | ¬ LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} ≤ δ :=
  probReal_textbookMixtureUpTo_failure_le_of_textbookGaussianPriorInput
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
    (n := n) (P := P) h hν hδ_pos hreg_pos
    (textbookGaussianPriorInput_of_designEigenvectorIntegral_supermartingale
      (A := A) (R := R) (ν := ν) (reg := reg) (β := β) (x := x) (P := P)
      h hν hreg_pos hσ2_pos)

/-- High-probability textbook mixture bound after discharging the anisotropic determinant integral
by spectral diagonalization of the LinUCB design matrix. -/
lemma probReal_textbookMixtureUpTo_ge_of_designEigenvectorIntegral
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hδ_pos : 0 < δ) (hreg_pos : 0 < reg) (hσ2_pos : 0 < (σ2 : ℝ)) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω} :=
  probReal_event_ge_of_failure_le (P := P)
    (probReal_textbookMixtureUpTo_failure_le_of_designEigenvectorIntegral
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
      (n := n) (P := P) h hν hδ_pos hreg_pos hσ2_pos)

/-- Failure-probability self-normalized noise bound from the named Gaussian-mixture input. -/
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

/-- High-probability self-normalized noise bound from the named Gaussian-mixture input. -/
lemma probReal_textbookNoise_ge_of_textbookGaussianMixtureInput
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    (μlambda : Measure (Feature d)) [SFinite μlambda]
    (h_mix : TextbookGaussianMixtureInput A R ν reg σ2 x P μlambda) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} :=
  probReal_textbookSelfNormalizedNoiseEventUpTo_ge_of_mixtureUpTo_ge
    (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
    (n := n) (P := P) hσ2_pos hδ_pos hreg_pos
    (probReal_textbookMixtureUpTo_ge_of_textbookGaussianMixtureInput
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
      (n := n) (P := P) h hν hδ_pos hreg_pos μlambda h_mix)

/-- Failure-probability self-normalized noise bound from the concrete Gaussian direction prior used
in the textbook method-of-mixtures proof. -/
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

/-- Failure-probability textbook self-normalized noise bound from the remaining unshifted
anisotropic Gaussian determinant integral. -/
lemma probReal_textbookNoise_failure_le_of_anisotropicKernelIntegral
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    (h_kernel_pos : ∀ i, i ≠ 0 → ∀ᵐ ω ∂P,
      Real.rpow ((σ2 : ℝ) * reg / (2 * Real.pi)) ((d : ℝ) / 2) *
          (∫ lambda, anisotropicGaussianKernel A reg σ2 x i ω lambda) =
        1 / √(designDetRatio A reg x i ω))
    (h_prod_pos : ∀ i, i ≠ 0 →
      Integrable
        (Function.uncurry fun ω lambda ↦
          Real.exp (centeredResponseDirectionalExponent A R ν reg σ2 x i ω lambda))
        (P.prod (gaussianDirectionMeasure d reg σ2))) :
    P.real {ω | ¬ LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} ≤
      δ :=
  probReal_textbookNoise_failure_le_of_textbookGaussianPriorInput
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
    (n := n) (P := P) h hν hσ2_pos hδ_pos hreg_pos
    (textbookGaussianPriorInput_of_anisotropicKernelIntegral_posTime
      (A := A) (R := R) (ν := ν) (reg := reg) (x := x) (P := P)
      hreg_pos hσ2_pos h_kernel_pos h_prod_pos)

/-- High-probability textbook self-normalized noise bound from the remaining unshifted anisotropic
Gaussian determinant integral. -/
lemma probReal_textbookNoise_ge_of_anisotropicKernelIntegral
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    (h_kernel_pos : ∀ i, i ≠ 0 → ∀ᵐ ω ∂P,
      Real.rpow ((σ2 : ℝ) * reg / (2 * Real.pi)) ((d : ℝ) / 2) *
          (∫ lambda, anisotropicGaussianKernel A reg σ2 x i ω lambda) =
        1 / √(designDetRatio A reg x i ω))
    (h_prod_pos : ∀ i, i ≠ 0 →
      Integrable
        (Function.uncurry fun ω lambda ↦
          Real.exp (centeredResponseDirectionalExponent A R ν reg σ2 x i ω lambda))
        (P.prod (gaussianDirectionMeasure d reg σ2))) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} :=
  probReal_textbookNoise_ge_of_textbookGaussianPriorInput
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
    (n := n) (P := P) h hν hσ2_pos hδ_pos hreg_pos
    (textbookGaussianPriorInput_of_anisotropicKernelIntegral_posTime
      (A := A) (R := R) (ν := ν) (reg := reg) (x := x) (P := P)
      hreg_pos hσ2_pos h_kernel_pos h_prod_pos)

/-- Failure-probability textbook self-normalized noise bound after discharging the anisotropic
determinant integral by spectral diagonalization of the LinUCB design matrix. -/
lemma probReal_textbookNoise_failure_le_of_designEigenvectorIntegral
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg) :
    P.real {ω | ¬ LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} ≤
      δ :=
  probReal_textbookNoise_failure_le_of_textbookGaussianPriorInput
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
    (n := n) (P := P) h hν hσ2_pos hδ_pos hreg_pos
    (textbookGaussianPriorInput_of_designEigenvectorIntegral_supermartingale
      (A := A) (R := R) (ν := ν) (reg := reg) (β := β) (x := x) (P := P)
      h hν hreg_pos hσ2_pos)

/-- High-probability textbook self-normalized noise bound after discharging the anisotropic
determinant integral by spectral diagonalization of the LinUCB design matrix. -/
lemma probReal_textbookNoise_ge_of_designEigenvectorIntegral
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    {σ2 : ℝ≥0} {δ : ℝ}
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} :=
  probReal_textbookNoise_ge_of_textbookGaussianPriorInput
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν)
    (n := n) (P := P) h hν hσ2_pos hδ_pos hreg_pos
    (textbookGaussianPriorInput_of_designEigenvectorIntegral_supermartingale
      (A := A) (R := R) (ν := ν) (reg := reg) (β := β) (x := x) (P := P)
      h hν hreg_pos hσ2_pos)

omit [IsMarkovKernel ν] in
/-- Failure-probability self-normalized noise bound obtained from the stopped
Gaussian-mixture statistic. -/
lemma probReal_textbookSelfNormalizedNoiseEventUpTo_failure_le_of_stoppedMixture_integral_le
    {σ2 : ℝ≥0} {δ : ℝ}
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    (hstop_integrable :
      Integrable (fun ω ↦ stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω) P)
    (hstop_integral :
      (∫ ω, stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω ∂P) ≤ 1) :
    P.real {ω | ¬ LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} ≤ δ :=
  probReal_textbookSelfNormalizedNoiseEventUpTo_failure_le_of_mixtureUpTo_failure_le
    (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
    (n := n) (P := P) hσ2_pos hδ_pos hreg_pos
    (probReal_textbookMixtureUpTo_failure_le_of_stoppedMixture_integral_le
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
      (n := n) (P := P) hδ_pos hstop_integrable hstop_integral)

omit [IsMarkovKernel ν] in
/-- High-probability self-normalized noise bound obtained from the stopped Gaussian-mixture
statistic. -/
lemma probReal_textbookSelfNormalizedNoiseEventUpTo_ge_of_stoppedMixture_integral_le
    {σ2 : ℝ≥0} {δ : ℝ}
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    (hstop_integrable :
      Integrable (fun ω ↦ stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω) P)
    (hstop_integral :
      (∫ ω, stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω ∂P) ≤ 1) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} :=
  probReal_textbookSelfNormalizedNoiseEventUpTo_ge_of_mixtureUpTo_ge
    (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
    (n := n) (P := P) hσ2_pos hδ_pos hreg_pos
    (probReal_textbookMixtureUpTo_ge_of_stoppedMixture_integral_le
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
      (n := n) (P := P) hδ_pos hstop_integrable hstop_integral)

omit [IsMarkovKernel ν] in
/-- Failure-probability self-normalized noise bound obtained from the bounded stopped
Gaussian-mixture supermartingale. -/
lemma probReal_textbookNoise_failure_le_of_boundedStoppedMixture_supermartingale
    {σ2 : ℝ≥0} {δ : ℝ}
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    {ℱ : Filtration ℕ mΩ} [SigmaFiniteFiltration P ℱ]
    (hM :
      Supermartingale
        (fun t ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν t ω) ℱ P) :
    P.real {ω | ¬ LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} ≤
      δ :=
  probReal_textbookSelfNormalizedNoiseEventUpTo_failure_le_of_mixtureUpTo_failure_le
    (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
    (n := n) (P := P) hσ2_pos hδ_pos hreg_pos
    (probReal_textbookMixtureUpTo_failure_le_of_boundedStoppedMixture_supermartingale
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
      (n := n) (P := P) hδ_pos hreg_pos (ℱ := ℱ) hM)

omit [IsMarkovKernel ν] in
/-- High-probability self-normalized noise bound obtained from the bounded stopped
Gaussian-mixture supermartingale. -/
lemma probReal_textbookNoise_ge_of_boundedStoppedMixture_supermartingale
    {σ2 : ℝ≥0} {δ : ℝ}
    (hσ2_pos : 0 < (σ2 : ℝ)) (hδ_pos : 0 < δ) (hreg_pos : 0 < reg)
    {ℱ : Filtration ℕ mΩ} [SigmaFiniteFiltration P ℱ]
    (hM :
      Supermartingale
        (fun t ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν t ω) ℱ P) :
    1 - δ ≤
      P.real {ω | LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} :=
  probReal_textbookSelfNormalizedNoiseEventUpTo_ge_of_mixtureUpTo_ge
    (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
    (n := n) (P := P) hσ2_pos hδ_pos hreg_pos
    (probReal_textbookMixtureUpTo_ge_of_boundedStoppedMixture_supermartingale
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
      (n := n) (P := P) hδ_pos hreg_pos (ℱ := ℱ) hM)

omit [IsMarkovKernel ν] in
/-- Probability monotonicity from the textbook determinant-ratio self-normalized event to the
centered-noise-plus-bias event. -/
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
