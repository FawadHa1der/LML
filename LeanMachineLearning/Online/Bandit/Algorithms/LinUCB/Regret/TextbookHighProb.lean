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

/-- High-probability textbook regret bound from the concrete Gaussian prior input. -/
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

/-- High-probability textbook regret bound after discharging the Gaussian-prior input by the
design-eigenvector integral route. -/
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

end LinUCB

end Bandits
