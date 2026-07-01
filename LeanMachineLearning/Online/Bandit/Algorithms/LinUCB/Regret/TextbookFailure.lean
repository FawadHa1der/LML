/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: OpenAI, Fawad Haider
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.LinUCB.Regret.TextbookHighProb

/-!
# LinUCB Regret: Textbook Failure-Probability Bounds

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

/-- Failure-probability wrapper for the deterministic textbook finite-action LinUCB regret bound,
consuming only a finite-horizon confidence event. -/
lemma probReal_textbook_regret_bound_deterministic_failure_le_of_confidenceEventUpTo_failure_le
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) {δ : ℝ}
    (h_conf_failure :
      P.real {ω | ¬ LinUCBConfidenceEventUpTo A R reg β x ν n ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ := by
  refine le_trans ?_ h_conf_failure
  refine probReal_failure_le_of_ae_imp (P := P) ?_
  filter_upwards [regret_ae_imp_le_textbook_finite_action_deterministic_bound_upTo
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2] with ω h_regret h_confω
  exact h_regret h_confω

/-- Short exported failure-probability endpoint for the finite-action LinUCB regret proof.

This is the failure-probability counterpart of `regret_bound_ge_of_confidenceEventUpTo_ge`. -/
-- ANCHOR: LinUCB.regret_bound_failure_le
lemma regret_bound_failure_le_of_confidenceEventUpTo_failure_le
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) {δ : ℝ}
    (h_conf_failure :
      P.real {ω | ¬ LinUCBConfidenceEventUpTo A R reg β x ν n ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ := by
  exact probReal_textbook_regret_bound_deterministic_failure_le_of_confidenceEventUpTo_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2 h_conf_failure
-- ANCHOR_END: LinUCB.regret_bound_failure_le

/-- Failure-probability textbook regret bound from the self-normalized prediction-confidence
event. -/
lemma probReal_textbook_regret_bound_deterministic_failure_le_of_selfNormalizedUpTo_failure_le
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2) {δ : ℝ}
    (h_self_failure :
      P.real {ω | ¬ LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ := by
  exact probReal_textbook_regret_bound_deterministic_failure_le_of_confidenceEventUpTo_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2
    (probReal_confidenceEventUpTo_failure_le_of_selfNormalizedConfidenceEventUpTo_failure_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      (P := P) h_self_failure)

lemma probReal_textbook_regret_bound_deterministic_failure_le_of_centeredNoiseBiasUpTo_failure_le
    [Nonempty (Fin K)]
    (h : IsAlgEnvSeq A R (linUCBAlgorithm hK reg β x) (stationaryEnv ν) P)
    (h_mean_bound : MeanRewardBound (K := K) ν (-1) 1)
    (hβ_schedule : BetaSchedule β)
    (hreg_pos : 0 < reg)
    (L2 : ℝ) (hL2 : FeatureSqNormBound x L2)
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    {δ : ℝ}
    (h_noise_failure :
      P.real {ω | ¬ LinUCBCenteredNoiseBiasConfidenceEventUpTo A R reg β x ν θ n ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ := by
  exact probReal_textbook_regret_bound_deterministic_failure_le_of_selfNormalizedUpTo_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2
    (probReal_selfNormalizedConfidenceEventUpTo_failure_le_of_centeredNoiseBias_failure_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      (P := P) θ h_linear hreg_pos h_noise_failure)

/-- Failure-probability deterministic LinUCB regret bound, consuming the textbook determinant-ratio
self-normalized centered-noise event directly. -/
lemma probReal_textbookRegretDet_failure_le_of_textbookNoiseUpTo_failure_le
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
    {σ2 : ℝ≥0} {δ : ℝ}
    (h_budget : ∀ᵐ ω ∂P, ∀ t, t ∈ range n → t ≠ 0 →
      (√(textbookSelfNormalizedNoiseBound σ2 δ (designDetRatio A reg x t ω)) +
        √(reg * S2)) ^ 2 ≤ β (t + 1))
    (h_noise_failure :
      P.real {ω |
        ¬ LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) + textbookRegretBonus (d := d) reg β L2 n} ≤ δ := by
  exact probReal_textbook_regret_bound_deterministic_failure_le_of_centeredNoiseBiasUpTo_failure_le
    (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n) h
    h_mean_bound hβ_schedule hreg_pos L2 hL2 θ h_linear
    (probReal_centeredNoiseBiasUpTo_failure_le_of_textbookNoiseUpTo_failure_le
      (A := A) (R := R) (reg := reg) (β := β) (x := x) (ν := ν) (n := n)
      (P := P) θ S2 hreg_pos hθ h_budget h_noise_failure)

/-- Failure-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook
beta schedule.

This is the failure-probability counterpart of
`probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_noiseUpTo_ge`. The beta schedule and
deterministic beta-radius budget are discharged internally; the only remaining probabilistic input
is the failure bound for the textbook self-normalized noise event. -/
lemma probReal_textbookRegretDet_failure_le_of_textbookLinUCBBeta_noiseUpTo_failure_le
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
    (h_noise_failure :
      P.real {ω |
        ¬ LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} ≤ δ := by
  simpa [textbookRegretBonus_textbookLinUCBBeta_eq] using
    probReal_textbookRegretDet_failure_le_of_textbookNoiseUpTo_failure_le
    (A := A) (R := R) (reg := reg)
    (β := textbookLinUCBBeta d reg S2 σ2 L2 n δ)
    (x := x) (ν := ν) (n := n) h h_mean_bound
    (textbookLinUCBBeta_schedule d reg S2 σ2 L2 n δ) hreg_pos L2 hL2 θ h_linear
    S2 hθ
    (textbookLinUCBBeta_budget_ae_of_featureSqNorm_bound (A := A) (reg := reg)
      (x := x) (n := n) (P := P) S2 L2 hreg_pos hd hL2 hδ_pos)
    h_noise_failure

/-- Failure-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook
beta schedule, with the bounded-mean assumption derived from linear realizability and normalized
feature/parameter norm bounds. -/
lemma probReal_textbookRegretDet_failure_le_of_textbookLinUCBBeta_noiseUpTo_linear
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
    (h_noise_failure :
      P.real {ω |
        ¬ LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω} ≤ δ) :
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} ≤ δ := by
  exact probReal_textbookRegretDet_failure_le_of_textbookLinUCBBeta_noiseUpTo_failure_le
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    (meanRewardBound_neg_one_one_of_linear_sq_norm_bounds
      (ν := ν) (x := x) (θ := θ) h_linear hL2 hθ hLS_le_one)
    hreg_pos hd hL2 θ h_linear hθ hδ_pos h_noise_failure

lemma probReal_textbookRegretDet_failure_le_of_textbookGaussianPrior_linear_all
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
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} ≤ δ := by
  by_cases hd : d = 0
  · exact probReal_failure_le_of_forall (P := P)
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
      hδ_pos.le
  · exact probReal_textbookRegretDet_failure_le_of_textbookLinUCBBeta_noiseUpTo_linear
      (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
      hreg_pos hd hL2 θ h_linear hθ hLS_le_one hδ_pos
      (probReal_textbookNoise_failure_le_of_textbookGaussianPriorInput
        (A := A) (R := R) (reg := reg)
        (β := textbookLinUCBBeta d reg S2 σ2 L2 n δ)
        (x := x) (ν := ν) (n := n) (P := P)
        h hν hσ2_pos hδ_pos hreg_pos h_prior)

/-- Failure-probability deterministic LinUCB regret bound for the fixed finite-horizon textbook
beta schedule, consuming the remaining unshifted anisotropic Gaussian determinant integral
directly. The determinant-integral identity is required only at positive times; time zero is
handled by the isotropic base case, and the product-integrability condition is likewise required
theorem also discharges the concrete Gaussian-prior product-integrability obligation from the
proved fixed-direction scalar supermartingale bound. -/
lemma probReal_textbookRegretDet_failure_le_of_designEigenvectorIntegral_linear_all
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
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} ≤ δ := by
  exact probReal_textbookRegretDet_failure_le_of_textbookGaussianPrior_linear_all
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    hreg_pos hL2 θ h_linear hθ hLS_le_one hσ2_pos hδ_pos hν
    (textbookGaussianPriorInput_of_designEigenvectorIntegral_supermartingale
      (A := A) (R := R) (ν := ν) (reg := reg)
      (β := textbookLinUCBBeta d reg S2 σ2 L2 n δ) (x := x) (P := P)
      h hν hreg_pos hσ2_pos)

/-- Final failure-probability finite-action LinUCB regret theorem for the textbook beta schedule.

This is the failure-probability counterpart of
`probReal_textbookRegretDet_ge_of_textbookLinUCBBeta_linear`: the spectral Gaussian integral,
product-integrability, and self-normalized Gaussian-mixture proof are discharged internally. -/
lemma probReal_textbookRegretDet_failure_le_of_textbookLinUCBBeta_linear
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
    P.real {ω |
      ¬ regret ν A n ω ≤
          (if n = 0 then 0 else 2) +
            textbookLinUCBRegretBonus d reg S2 σ2 L2 n δ} ≤ δ :=
  probReal_textbookRegretDet_failure_le_of_designEigenvectorIntegral_linear_all
    (A := A) (R := R) (reg := reg) (x := x) (ν := ν) (n := n) S2 L2 h
    hreg_pos hL2 θ h_linear hθ hLS_le_one hσ2_pos hδ_pos hν

end LinUCB

end Bandits
