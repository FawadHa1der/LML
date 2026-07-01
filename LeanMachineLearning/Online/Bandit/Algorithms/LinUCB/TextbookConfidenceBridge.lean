/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: OpenAI, Fawad Haider
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.LinUCB.TextbookMixture

/-!
# LinUCB Textbook Confidence Bridge

Deterministic bridges from centered-noise and textbook self-normalized events to
LinUCB prediction-confidence events.
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

omit [IsMarkovKernel ν] [IsProbabilityMeasure P] in
/-- The textbook determinant-ratio self-normalized noise event plus the ridge-bias radius implies
the centered-noise-plus-bias confidence event. -/
lemma LinUCBCenteredNoiseBiasConfidenceEventUpTo.of_textbookSelfNormalizedNoise
    (θ : Feature d) (S2 : ℝ) {σ2 : ℝ≥0} {δ : ℝ}
    (hreg_pos : 0 < reg)
    (hθ : ParameterSqNormBound θ S2)
    (h_noise :
      LinUCBTextbookSelfNormalizedNoiseEventUpTo A R reg σ2 δ x ν n ω)
    (h_budget : ∀ t, t ∈ range n → t ≠ 0 →
      (√(textbookSelfNormalizedNoiseBound σ2 δ (designDetRatio A reg x t ω)) +
        √(reg * S2)) ^ 2 ≤ β (t + 1)) :
    LinUCBCenteredNoiseBiasConfidenceEventUpTo A R reg β x ν θ n ω := by
  intro t ht ht0
  exact
    (centeredNoiseBiasQuadraticForm_le_sqrt_bounds_sq (A := A) (R := R)
      (reg := reg) (x := x) (ν := ν) (n := t) (ω := ω) θ hreg_pos
      (textbookSelfNormalizedNoiseBound σ2 δ (designDetRatio A reg x t ω))
      (reg * S2) (h_noise t ht ht0)
      (regularizationBiasQuadraticForm_le_of_parameterSqNormBound (A := A)
        (reg := reg) (x := x) (n := t) (ω := ω) θ hreg_pos hθ)).trans
      (h_budget t ht ht0)

omit [IsMarkovKernel ν] in
lemma LinUCBParameterEllipsoidConfidenceEventUpTo.of_centeredNoiseBias
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg)
    (h_noise :
      LinUCBCenteredNoiseBiasConfidenceEventUpTo A R reg β x ν θ n ω) :
    LinUCBParameterEllipsoidConfidenceEventUpTo A R reg β x θ n ω := by
  intro t ht ht0
  rw [parameterErrorQuadraticForm_eq_centeredNoiseBiasQuadraticForm (A := A) (R := R)
    (reg := reg) (x := x) (ν := ν) (n := t) (ω := ω) θ h_linear hreg_pos]
  exact h_noise t ht ht0

/-- Horizon-local self-normalized prediction-confidence event for all finite actions. -/
def LinUCBSelfNormalizedConfidenceEventUpTo
    (A : ℕ → Ω → Fin K) (R : ℕ → Ω → ℝ)
    (reg : ℝ) (β : ℕ → ℝ) (x : Fin K → Feature d)
    (ν : Kernel (Fin K) ℝ) (n : ℕ) (ω : Ω) : Prop :=
  ∀ t, t ∈ range n → t ≠ 0 → ∀ a,
    |estimatedReward A R reg x a t ω - (ν a)[id]| ≤
      √(β (t + 1)) * width A reg x a t ω

omit [IsMarkovKernel ν] in
lemma LinUCBSelfNormalizedConfidenceEventUpTo.of_parameterEllipsoid
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg)
    (h_ellipsoid : LinUCBParameterEllipsoidConfidenceEventUpTo A R reg β x θ n ω) :
    LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω := by
  intro t ht ht0 a
  have h_cauchy_t :=
    linUCBPredictionErrorCauchySchwarz_of_reg_pos (A := A) (reg := reg) (x := x)
      hreg_pos (thetaHat A R reg x t ω - θ) a t ω
  have h_radius :
      √(parameterErrorQuadraticForm A R reg x θ t ω) ≤ √(β (t + 1)) :=
    Real.sqrt_le_sqrt (h_ellipsoid t ht ht0)
  have h_error :
      estimatedReward A R reg x a t ω - (ν a)[id] =
        dotProduct (thetaHat A R reg x t ω - θ) (x a) := by
    calc
      estimatedReward A R reg x a t ω - (ν a)[id]
          = dotProduct (thetaHat A R reg x t ω) (x a) - dotProduct θ (x a) := by
              rw [estimatedReward, h_linear a]
      _ = dotProduct (thetaHat A R reg x t ω - θ) (x a) := by
              simp only [WithLp.ofLp_sub]
              rw [sub_dotProduct]
  rw [h_error]
  calc
    |dotProduct (thetaHat A R reg x t ω - θ) (x a)|
        ≤ √(parameterErrorQuadraticForm A R reg x θ t ω) * width A reg x a t ω := by
          simpa [parameterErrorQuadraticForm] using h_cauchy_t
    _ ≤ √(β (t + 1)) * width A reg x a t ω := by
          exact mul_le_mul_of_nonneg_right h_radius (Real.sqrt_nonneg _)

omit [IsMarkovKernel ν] in
/-- The horizon-local centered-noise-plus-bias event implies the horizon-local self-normalized
prediction-error event under linear realizability and positive regularization. -/
lemma LinUCBSelfNormalizedConfidenceEventUpTo.of_centeredNoiseBias
    (θ : Feature d)
    (h_linear : LinearMeanModel ν x θ)
    (hreg_pos : 0 < reg)
    (h_noise : LinUCBCenteredNoiseBiasConfidenceEventUpTo A R reg β x ν θ n ω) :
    LinUCBSelfNormalizedConfidenceEventUpTo A R reg β x ν n ω :=
  LinUCBSelfNormalizedConfidenceEventUpTo.of_parameterEllipsoid (A := A) (R := R)
    (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (ω := ω) θ h_linear hreg_pos
    (LinUCBParameterEllipsoidConfidenceEventUpTo.of_centeredNoiseBias (A := A) (R := R)
      (reg := reg) (β := β) (x := x) (ν := ν) (n := n) (ω := ω)
      θ h_linear hreg_pos h_noise)

end AlgorithmBehavior

end LinUCB

end Bandits
