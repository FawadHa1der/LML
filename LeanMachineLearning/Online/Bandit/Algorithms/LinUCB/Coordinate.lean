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
  have hL2_nonneg : 0 ≤ L2 := (sq_nonneg ‖x a‖).trans (hL2 a)
  have hdot :
      |dotProduct θ (x a)| ≤ √S2 * √L2 :=
    abs_dotProduct_le_sqrt_mul_sqrt_of_sq_norm_le (u := θ) (v := x a)
      (U := S2) (V := L2) hθ (by simpa [dotProduct_self_eq_norm_sq] using hL2 a)
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

lemma width_eq_width' (reg : ℝ) (x : Fin K → Feature d)
    (a : Fin K) (n : ℕ) (ω : Ω) (hn : n ≠ 0) :
    width A reg x a n ω = width' reg x (n - 1) (IsAlgEnvSeq.hist A R (n - 1) ω) a := by
  simp [width, width', widthQuadraticForm_eq_widthQuadraticForm' (A := A) (R := R) reg x a n
    ω hn]

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


end AlgorithmBehavior

end LinUCB

end Bandits
