/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: OpenAI, Fawad Haider
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.LinUCB.Regret.Textbook

/-!
# LinUCB Regret: Textbook High-Probability Bounds

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

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule.

Compared with `probReal_textbook_regret_bound_deterministic_ge_of_textbookNoiseUpTo_ge`, this
lemma discharges the beta schedule and beta-radius budget internally from the bounded-feature
assumption and the determinant/trace comparison. The remaining probabilistic input is exactly the
textbook determinant-ratio self-normalized noise event. -/
lemma probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_noiseUpTo_ge
    [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hreg_pos : 0 < reg)
    (hd : d ≠ 0)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hδ_pos : 0 < δ)
    (h_noise_prob :
      1 - δ ≤
        P.real {ω |
          LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  simpa [textbookRegretBonus_textbookLinUCBBeta_eq] using
    probReal_textbook_regret_bound_deterministic_ge_of_textbookNoiseUpTo_ge
    (A := A) (R := R) (reg := reg)
    (β := textbookLinUCBBeta d reg S2 σ2 L2 n δ)
    (x := x) (ν := ν) (n := n) h h_mean_bound
    (textbookLinUCBBeta_schedule d reg S2 σ2 L2 n δ) hreg_pos L2 hL2 θ h_linear
    S2 hθ
    (textbookLinUCBBeta_budget_ae_of_featureSqNorm_bound (A := A) (reg := reg)
      (x := x) (n := n) (P := P) S2 L2 hreg_pos hd hL2 hδ_pos)
    h_noise_prob

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, with the bounded-mean assumption derived from linear realizability and normalized
feature/parameter norm bounds. -/
lemma probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_noiseUpTo_ge_of_linear_sq_norm_bounds
    [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hd : d ≠ 0)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hδ_pos : 0 < δ)
    (h_noise_prob :
      1 - δ ≤
        P.real {ω |
          LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  exact probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_noiseUpTo_ge
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    (meanRewardBound_neg_one_one_of_linear_sq_norm_bounds
    (ν := ν) (x := x) (θ := θ) h_linear hL2 hθ hLS_le_one)
    hreg_pos hd hL2 θ h_linear hθ hδ_pos h_noise_prob

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, consuming the Gaussian-mixture event that appears in the textbook proof.

Compared with `probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_noiseUpTo_ge`, this theorem
moves the probabilistic input one step earlier in the textbook argument: a future Gaussian-mixture
concentration theorem should prove `LinUCBTextbookMixtureBoundEventUpTo`, and the deterministic
bridge above converts it into the textbook self-normalized noise event. -/
lemma probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_mixtureUpTo_ge
    [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hreg_pos : 0 < reg)
    (hd : d ≠ 0)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    (h_mix_prob :
      1 - δ ≤
        P.real {ω |
          LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  exact probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_noiseUpTo_ge
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    h_mean_bound hreg_pos hd hL2 θ h_linear hθ hδ_pos
    (probReal_textbookSelfNormalizedNoiseEventUpTo_ge_of_mixtureUpTo_ge
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
      (n := n) (P := P) hσ2_pos hδ_pos hreg_pos h_mix_prob)

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, consuming the Gaussian-mixture event and deriving the bounded-mean assumption from the
linear model and squared-norm bounds. -/
lemma probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_mixtureUpTo_linear
    [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hd : d ≠ 0)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    (h_mix_prob :
      1 - δ ≤
        P.real {ω |
          LinUCBTextbookMixtureBoundEventUpTo A R reg σ2 δ x ν n ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  exact probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_mixtureUpTo_ge
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    (meanRewardBound_neg_one_one_of_linear_sq_norm_bounds
      (ν := ν) (x := x) (θ := θ) h_linear hL2 hθ hLS_le_one)
    hreg_pos hd hL2 θ h_linear hθ hσ2_pos hδ_pos h_mix_prob

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, consuming the stopped Gaussian-mixture statistic expected by the textbook
self-normalized proof.

The remaining analytic concentration theorem should prove the two stopped-statistic inputs here:
integrability and expectation at most one. -/
lemma probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_stoppedMixture_integral_le
    [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hreg_pos : 0 < reg)
    (hd : d ≠ 0)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    (hstop_integrable :
      Integrable (fun ω ↦ stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω) P)
    (hstop_integral :
      (∫ ω, stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω ∂P) ≤ 1) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  exact probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_mixtureUpTo_ge
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    h_mean_bound hreg_pos hd hL2 θ h_linear hθ hσ2_pos hδ_pos
    (probReal_textbookMixtureUpTo_ge_of_stoppedMixture_integral_le
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
      (n := n) (P := P) hδ_pos hstop_integrable hstop_integral)

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, consuming the supermartingale/stopping-time form of the textbook Gaussian-mixture
argument. -/
lemma probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_stoppedMixture_supermartingale
    [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hreg_pos : 0 < reg)
    (hd : d ≠ 0)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    {ℱ : Filtration ℕ mΩ} [SigmaFiniteFiltration P ℱ]
    (hM :
      Supermartingale
        (fun t ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν t ω) ℱ P)
    (hτ :
      IsStoppingTime ℱ
        (fun ω ↦ (firstTextbookMixtureFailureTime A R reg σ2 δ x ν n ω : ℕ∞))) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  exact probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_stoppedMixture_integral_le
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    h_mean_bound hreg_pos hd hL2 θ h_linear hθ hσ2_pos hδ_pos
    (integrable_stoppedTextbookMixtureStatistic_of_supermartingale
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x)
      (ν := ν) (n := n) (P := P) (ℱ := ℱ) hM hτ)
    (integral_stoppedTextbookMixtureStatistic_le_one_of_supermartingale
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x)
      (ν := ν) (n := n) (P := P) (ℱ := ℱ) hM hτ hreg_pos)

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, consuming the bounded-stopped supermartingale form of the textbook Gaussian-mixture
argument.

This is the optional-stopping-friendly version of
`probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_stoppedMixture_supermartingale`: the
bounded first-crossing time is built into the theorem, so the caller only supplies the
supermartingale property of the textbook mixture statistic. -/
lemma probReal_textbookRegretDet_ge_of_boundedStoppedMixture_supermartingale
    [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hreg_pos : 0 < reg)
    (hd : d ≠ 0)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    {ℱ : Filtration ℕ mΩ} [SigmaFiniteFiltration P ℱ]
    (hM :
      Supermartingale
        (fun t ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν t ω) ℱ P) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  exact probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_mixtureUpTo_ge
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    h_mean_bound hreg_pos hd hL2 θ h_linear hθ hσ2_pos hδ_pos
    (probReal_textbookMixtureUpTo_ge_of_boundedStoppedMixture_supermartingale
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν)
      (n := n) (P := P) hδ_pos hreg_pos (ℱ := ℱ) hM)

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, consuming the stopped Gaussian-mixture statistic and deriving the bounded-mean
assumption from linear realizability and normalized squared-norm bounds. -/
lemma probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_stoppedMixture_linear
    [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hd : d ≠ 0)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    (hstop_integrable :
      Integrable (fun ω ↦ stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω) P)
    (hstop_integral :
      (∫ ω, stoppedTextbookMixtureStatistic A R reg σ2 δ x ν n ω ∂P) ≤ 1) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  exact probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_stoppedMixture_integral_le
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    (meanRewardBound_neg_one_one_of_linear_sq_norm_bounds
      (ν := ν) (x := x) (θ := θ) h_linear hL2 hθ hLS_le_one)
    hreg_pos hd hL2 θ h_linear hθ hσ2_pos hδ_pos hstop_integrable hstop_integral

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, with the bounded-mean assumption derived from linear realizability and normalized
feature/parameter norm bounds, consuming only the bounded-stopped mixture supermartingale. -/
lemma probReal_textbookRegretDet_ge_of_boundedStoppedMixture_linear
    [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hd : d ≠ 0)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    {ℱ : Filtration ℕ mΩ} [SigmaFiniteFiltration P ℱ]
    (hM :
      Supermartingale
        (fun t ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν t ω) ℱ P) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  exact probReal_textbookRegretDet_ge_of_boundedStoppedMixture_supermartingale
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    (meanRewardBound_neg_one_one_of_linear_sq_norm_bounds
      (ν := ν) (x := x) (θ := θ) h_linear hL2 hθ hLS_le_one)
    hreg_pos hd hL2 θ h_linear hθ hσ2_pos hδ_pos (ℱ := ℱ) hM

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, consuming only the bounded-stopped mixture supermartingale and handling the
zero-dimensional case internally. -/
lemma probReal_textbookRegretDet_ge_of_boundedStoppedMixture_linear_all
    [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    {ℱ : Filtration ℕ mΩ} [SigmaFiniteFiltration P ℱ]
    (hM :
      Supermartingale
        (fun t ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν t ω) ℱ P) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  by_cases hd : d = 0
  · exact probReal_event_ge_of_failure_le (P := P)
      (probReal_failure_le_of_forall (P := P)
        (F := fun ω ↦
          regret ν A n ω ≤
            (if n = 0 then 0 else 2) +
              textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ)
        (fun ω ↦ by
          simpa [textbookRegretBonus_textbookLinUCBBeta_eq] using
            regret_le_textbookRegretDet_bound_of_linear_dim_eq_zero
              (A := A) (ν := ν) (reg := reg)
              (β := textbookLinUCBBeta d reg S2 σ2 L2 n δ)
              (x := x) (n := n) (ω := ω) L2 (θ := θ) hd h_linear)
        hδ_pos.le)
  · exact probReal_textbookRegretDet_ge_of_boundedStoppedMixture_linear
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
      hreg_pos hd hL2 θ h_linear hθ hLS_le_one hσ2_pos hδ_pos (ℱ := ℱ) hM

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, consuming the directional Gaussian-mixture identity.

This theorem is one step closer to the textbook proof than
`probReal_textbookRegretDet_ge_of_boundedStoppedMixture_linear_all`: it no longer asks the caller
to supply the mixture statistic as a supermartingale. Instead, it builds that supermartingale from
the proved fixed-direction scalar concentration theorem plus the remaining analytic
method-of-mixtures inputs:

* integrability of the closed-form mixture statistic;
* integrability of the direction-indexed set integrals;
* the set-integral identity identifying the closed-form determinant-ratio statistic with the
  integral of the fixed-direction exponential process over directions.

The last item is the substantive multivariate Gaussian integral still needed to remove the
remaining assumptions and match the textbook theorem fully. -/
lemma probReal_textbookRegretDet_ge_of_directionalMixture_linear_all
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (μlambda : Measure (Feature d))
    (h_int_stat : ∀ i,
      Integrable (fun ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν i ω) P)
    (h_inner_int : ∀ i, ∀ s : Set Ω, MeasurableSet s →
      Integrable
        (fun lambda ↦
          ∫ ω in s,
            Real.exp (centeredResponseDirectionalExponent A R ν reg σ2 x i ω lambda) ∂P)
        μlambda)
    (h_mix_set : ∀ i, ∀ s : Set Ω, MeasurableSet s →
      (∫ ω in s, textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν i ω ∂P) =
        ∫ lambda,
          (∫ ω in s,
            Real.exp (centeredResponseDirectionalExponent A R ν reg σ2 x i ω lambda) ∂P)
          ∂μlambda) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  let ℱ := IsAlgEnvSeq.filtrationAction h.measurable_action h.measurable_feedback
  have hM :
      Supermartingale
        (fun t ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν t ω) ℱ P :=
    supermartingale_textbookSelfNormalizedMixtureStatistic_filtrationAction_of_directionalMixture
      (A := A) (R := R) (reg := reg)
      (β := textbookLinUCBBeta d reg S2 σ2 L2 n δ) (x := x) (ν := ν)
      h hν μlambda h_int_stat h_inner_int h_mix_set
  exact probReal_textbookRegretDet_ge_of_boundedStoppedMixture_linear_all
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    hreg_pos hL2 θ h_linear hθ hLS_le_one hσ2_pos hδ_pos (ℱ := ℱ) hM

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, consuming a pointwise Gaussian-mixture identity.

This is the regret-level interface closest to the textbook method-of-mixtures proof among the
currently proved wrappers. It reuses
`probReal_textbookRegretDet_ge_of_directionalMixture_linear_all`, but obtains that theorem's
set-integral mixture identity from a pointwise determinant-ratio identity plus an explicit
Fubini/Tonelli swap. -/
lemma probReal_textbookRegretDet_ge_of_directionalMixture_pointwise_linear_all
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (μlambda : Measure (Feature d))
    (h_int_stat : ∀ i,
      Integrable (fun ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν i ω) P)
    (h_inner_int : ∀ i, ∀ s : Set Ω, MeasurableSet s →
      Integrable
        (fun lambda ↦
          ∫ ω in s,
            Real.exp (centeredResponseDirectionalExponent A R ν reg σ2 x i ω lambda) ∂P)
        μlambda)
    (h_pointwise : ∀ i,
      (fun ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν i ω) =ᵐ[P]
        fun ω ↦
          ∫ lambda,
            Real.exp (centeredResponseDirectionalExponent A R ν reg σ2 x i ω lambda)
          ∂μlambda)
    (h_fubini : ∀ i, ∀ s : Set Ω, MeasurableSet s →
      (∫ ω in s,
        (∫ lambda,
          Real.exp (centeredResponseDirectionalExponent A R ν reg σ2 x i ω lambda)
        ∂μlambda) ∂P) =
        ∫ lambda,
          (∫ ω in s,
            Real.exp (centeredResponseDirectionalExponent A R ν reg σ2 x i ω lambda) ∂P)
          ∂μlambda) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  exact probReal_textbookRegretDet_ge_of_directionalMixture_linear_all
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    hreg_pos hL2 θ h_linear hθ hLS_le_one hσ2_pos hδ_pos hν μlambda
    h_int_stat h_inner_int
    (textbookSelfNormalizedMixtureStatistic_setIntegral_eq_integral_of_ae_eq_integral
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (x := x) (ν := ν) (P := P)
      μlambda h_pointwise h_fubini)

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, consuming a pointwise Gaussian-mixture identity plus product-integrability.

This removes the separate `h_int_stat`, `h_inner_int`, and `h_fubini` assumptions from
`probReal_textbookRegretDet_ge_of_directionalMixture_pointwise_linear_all`. They are all supplied
by mathlib's Fubini API from the single product-integrability hypothesis below. -/
lemma probReal_textbookRegretDet_ge_of_directionalMixture_prod_integrable_linear_all
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (μlambda : Measure (Feature d)) [SFinite μlambda]
    (h_pointwise : ∀ i,
      (fun ω ↦ textbookSelfNormalizedMixtureStatistic A R reg σ2 x ν i ω) =ᵐ[P]
        fun ω ↦
          ∫ lambda,
            Real.exp (centeredResponseDirectionalExponent A R ν reg σ2 x i ω lambda)
          ∂μlambda)
    (h_prod_int : ∀ i, ∀ s : Set Ω, MeasurableSet s →
      Integrable
        (Function.uncurry fun ω lambda ↦
          Real.exp (centeredResponseDirectionalExponent A R ν reg σ2 x i ω lambda))
        ((P.restrict s).prod μlambda)) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  exact probReal_textbookRegretDet_ge_of_directionalMixture_pointwise_linear_all
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    hreg_pos hL2 θ h_linear hθ hLS_le_one hσ2_pos hδ_pos hν μlambda
    (textbookMixtureStatistic_integrable_of_pointwise_prod_integrable
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (x := x) (ν := ν) (P := P)
      μlambda h_pointwise h_prod_int)
    (directionalMixture_inner_integrable_of_prod_integrable
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (x := x) (ν := ν) (P := P)
      μlambda h_prod_int)
    h_pointwise
    (directionalMixture_fubini_of_prod_integrable
      (A := A) (R := R) (reg := reg) (σ2 := σ2) (x := x) (ν := ν) (P := P)
      μlambda h_prod_int)

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, consuming the named textbook Gaussian-mixture input.

This is the lightest current mixture-based regret interface. The single `h_mix` hypothesis is the
remaining analytic method-of-mixtures theorem: for a concrete Gaussian direction measure it should
provide the pointwise determinant-ratio identity and global product-integrability. -/
lemma probReal_textbookRegretDet_ge_of_directionalMixture_global_prod_integrable_linear_all
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (μlambda : Measure (Feature d)) [SFinite μlambda]
    (h_mix : TextbookGaussianMixtureInput A R ν reg σ2 x P μlambda) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  exact probReal_textbookRegretDet_ge_of_directionalMixture_prod_integrable_linear_all
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    hreg_pos hL2 θ h_linear hθ hLS_le_one hσ2_pos hδ_pos hν μlambda h_mix.pointwise
    (fun i s _hs ↦
      directionalMixture_prod_integrable_of_global_prod_integrable
        (A := A) (R := R) (reg := reg) (σ2 := σ2) (x := x) (ν := ν) (P := P)
        μlambda h_mix.globalProdIntegrable i s)

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, using the concrete Gaussian direction prior from the textbook method of mixtures.

This removes the arbitrary direction measure from
`probReal_textbookRegretDet_ge_of_directionalMixture_global_prod_integrable_linear_all`. The only
remaining analytic input is the concrete Gaussian-prior identity/integrability package
`TextbookGaussianPriorInput`. -/
lemma probReal_textbookRegretDet_ge_of_textbookGaussianPrior_linear_all
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
    (h_prior : TextbookGaussianPriorInput A R ν reg σ2 x P) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  by_cases hd : d = 0
  · exact probReal_event_ge_of_failure_le (P := P)
      (probReal_failure_le_of_forall (P := P)
        (F := fun ω ↦
          regret ν A n ω ≤
            (if n = 0 then 0 else 2) +
              textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ)
        (fun ω ↦ by
          simpa [textbookRegretBonus_textbookLinUCBBeta_eq] using
            regret_le_textbookRegretDet_bound_of_linear_dim_eq_zero
              (A := A) (ν := ν) (reg := reg)
              (β := textbookLinUCBBeta d reg S2 σ2 L2 n δ)
              (x := x) (n := n) (ω := ω) L2 (θ := θ) hd h_linear)
        hδ_pos.le)
  · exact probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_noiseUpTo_ge_of_linear_sq_norm_bounds
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
      hreg_pos hd hL2 θ h_linear hθ hLS_le_one hδ_pos
      (probReal_textbookNoise_ge_of_textbookGaussianPriorInput
        (A := A) (R := R) (reg := reg)
        (β := textbookLinUCBBeta d reg S2 σ2 L2 n δ)
        (x := x) (ν := ν) (n := n) (P := P)
        h hν hσ2_pos hδ_pos hreg_pos h_prior)

/-- High-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook beta
schedule, consuming the remaining unshifted anisotropic Gaussian determinant integral directly.

Compared with `probReal_textbookRegretDet_ge_of_textbookGaussianPrior_linear_all`, this exposes the
actual analytic obligations left by the textbook method-of-mixtures proof: the unshifted Gaussian
determinant integral and product-integrability condition used for Fubini, both only at positive
times. The time-zero determinant-integral identity and integrability case are discharged
internally. -/
lemma probReal_textbookRegretDet_ge_of_anisotropicKernelIntegral_linear_all
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2)
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
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  exact probReal_textbookRegretDet_ge_of_textbookGaussianPrior_linear_all
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    hreg_pos hL2 θ h_linear hθ hLS_le_one hσ2_pos hδ_pos hν
    (textbookGaussianPriorInput_of_anisotropicKernelIntegral_posTime
      (A := A) (R := R) (ν := ν) (reg := reg) (x := x) (P := P)
      hreg_pos hσ2_pos h_kernel_pos h_prod_pos)

/-- High-probability deterministic LinUCB regret bound after discharging the anisotropic Gaussian
determinant integral by spectral diagonalization of the LinUCB design matrix.

Compared with `probReal_textbookRegretDet_ge_of_anisotropicKernelIntegral_linear_all`, this theorem
also discharges the concrete Gaussian-prior product-integrability obligation from the proved
fixed-direction scalar supermartingale bound. -/
lemma probReal_textbookRegretDet_ge_of_designEigenvectorIntegral_linear_all
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  exact probReal_textbookRegretDet_ge_of_textbookGaussianPrior_linear_all
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    hreg_pos hL2 θ h_linear hθ hLS_le_one hσ2_pos hδ_pos hν
    (textbookGaussianPriorInput_of_designEigenvectorIntegral_supermartingale
      (A := A) (R := R) (ν := ν) (reg := reg)
      (β := textbookLinUCBBeta d reg S2 σ2 L2 n δ) (x := x) (P := P)
      h hν hreg_pos hσ2_pos)

/-- Final high-probability finite-action LinUCB regret theorem for the textbook beta schedule.

This is the theorem-level wrapper: the self-normalized Gaussian-mixture proof, spectral Gaussian
integral, and product-integrability bookkeeping are all discharged internally. The remaining
assumptions are the standard finite-action linear-bandit model assumptions used by this
formalization: positive regularization/noise scale, squared feature and parameter bounds `L2` and
`S2`, linear means, normalized mean scale `L2 * S2 ≤ 1`, and subgaussian reward noise. The
`if n = 0 then 0 else 2` term is the finite-action bounded-gap bookkeeping used by this repo's
zero-indexed regret chain. -/
lemma probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_linear
    [StandardBorelSpace Ω] [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hσ2_pos : 0 < (σ2 : ℝ))
    (hδ_pos : 0 < δ)
    (hν : RewardNoiseSubgaussian (K := K) ν σ2) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} :=
  probReal_textbookRegretDet_ge_of_designEigenvectorIntegral_linear_all
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    hreg_pos hL2 θ h_linear hθ hLS_le_one hσ2_pos hδ_pos hν

/-- High-probability deterministic LinUCB regret bound for the trivial probability range
`δ ≥ 1`.

No concentration theorem is needed in this branch: `1 - δ ≤ 0`, so the textbook self-normalized
noise event automatically has probability at least `1 - δ`. The nontrivial future concentration
theorem only has to prove the case `δ < 1`. -/
lemma probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_one_le_delta_linear
    [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hd : d ≠ 0)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hδ_one : 1 ≤ δ) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  have hδ_pos : 0 < δ := by linarith
  exact probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_noiseUpTo_ge_of_linear_sq_norm_bounds
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    hreg_pos hd hL2 θ h_linear hθ hLS_le_one hδ_pos
    (probReal_textbookSelfNormalizedNoiseEventUpTo_ge_of_one_le_delta (A := A) (R := R)
      (reg := reg) (σ2 := σ2) (δ := δ) (x := x) (ν := ν) (n := n) (P := P)
      hδ_one)

/-- High-probability deterministic LinUCB regret bound when the future textbook self-normalized
concentration theorem is supplied only for the nontrivial range `δ < 1`.

The proof splits on `1 ≤ δ`. The `δ ≥ 1` branch is automatic, while the `δ < 1` branch consumes
the supplied textbook-noise probability bound. -/
lemma probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_nontrivial_noise_linear
    [Nonempty (Fin K)]
    {σ2 : ℝ≥0} {δ : ℝ}
    (S2 L2 : ℝ)
    (h : IsAlgEnvSeq A R
      (linUCBAlgorithm hK reg (textbookLinUCBBeta d reg S2 σ2 L2 n δ) x)
      (stationaryEnv ν) P)
    (hreg_pos : 0 < reg)
    (hd : d ≠ 0)
    (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hθ : ParameterSqNormBound θ S2)
    (hLS_le_one : L2 * S2 ≤ 1)
    (hδ_pos : 0 < δ)
    (h_noise_prob_lt_one : δ < 1 →
      1 - δ ≤
        P.real {ω |
          LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} := by
  by_cases hδ_one : 1 ≤ δ
  · exact probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_one_le_delta_linear
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
      hreg_pos hd hL2 θ h_linear hθ hLS_le_one hδ_one
  · exact probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_noiseUpTo_ge_of_linear_sq_norm_bounds
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
      hreg_pos hd hL2 θ h_linear hθ hLS_le_one hδ_pos
      (h_noise_prob_lt_one (lt_of_not_ge hδ_one))

/-- High-probability deterministic LinUCB regret bound, consuming the horizon-local
self-normalized centered-noise event plus the deterministic ridge-bias radius.

This is the closest current interface to the textbook confidence proof: a future vector
self-normalized concentration theorem should prove the displayed centered-noise event for a
concrete `noiseBudget`, and the radius condition packages the textbook choice of `β`. -/
lemma probReal_textbook_regret_bound_deterministic_ge_of_centeredNoiseConfidenceEventUpTo_ge
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
    (h_noise_prob :
      1 - δ ≤
        P.real {ω | LinUCBCenteredNoiseConfidenceEventUpTo A R reg noiseBudget x ν n ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  exact probReal_textbook_regret_bound_deterministic_ge_of_centeredNoiseBiasConfidenceEventUpTo_ge
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear
    (probReal_centeredNoiseBiasConfidenceEventUpTo_ge_of_centeredNoiseConfidenceEventUpTo_ge
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      (P := P) θ S2 hreg_pos hθ h_budget h_noise_prob)

/-- High-probability deterministic LinUCB regret bound, consuming a coordinate-wise
finite-horizon centered-noise event.

This is a conservative finite-dimensional route from scalar concentration lemmas into the existing
textbook-shaped regret theorem. It is looser than the exact self-normalized concentration theorem
because it first bounds the Euclidean norm coordinate-wise and then uses `V_t⁻¹ ≤ (reg I)⁻¹`. -/
lemma probReal_textbook_regret_bound_deterministic_ge_of_coordinateBoundEventUpTo_ge
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
    (hcoord_prob :
      1 - δ ≤
        P.real {ω |
          LinUCBCenteredNoiseCoordinateBoundEventUpTo A R coordBudget x ν n ω}) :
    1 - δ ≤
      P.real {ω |
        regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} := by
  exact probReal_textbook_regret_bound_deterministic_ge_of_centeredNoiseConfidenceEventUpTo_ge
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear S2 hθ h_noise_budget
    (probReal_centeredNoiseConfidenceEventUpTo_ge_of_coordinateBoundEventUpTo_ge
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) (P := P)
      hreg_pos hcoord_nonneg h_coord_budget hcoord_prob)


end LinUCB

end Bandits
