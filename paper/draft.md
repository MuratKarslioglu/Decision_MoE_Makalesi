# Does Expert Selection Matter? Multi-Criteria and Reinforcement-Learned Routing in a Sparsely Upcycled Vision-Language MoE

*Taslak (iskelet + gerçek sayılar). Tablolar `paper/tables/t1–t5.tex`, figürler `paper/figures/f2–f7`. Sayılar test setinden
(1321 soru) ve `experiments/results/` dosyalarındandır; işaretli yerler ek koşular bitince güncellenir.*

## Abstract
Mixture-of-Experts vision-language models route every token to a few experts, and a growing literature proposes smarter
routers. We ask a prior question: *does the choice of experts matter at all?* We build DecisionMoE, a router that ranks experts
with TOPSIS over seven criteria whose weights, and the number of active experts K, are chosen per step by a PPO policy,
and evaluate it on MoE-LLaVA-Phi2-2.7B×4 with 8,803 synthetic questions over five question types and with POPE.
(i) On this backbone, accuracy is insensitive to expert selection: Top-K for K=1..4, random routing, every single expert
alone and fixed MCDM routing all land in a 2.5-point band (0.380–0.405), because the experts of a sparsely upcycled model
differ by only 3–4% in weights. (ii) Top-1 matches Top-2 at half the FLOPs. (iii) A learned Dynamic-K policy discovers
this by itself — K→1 on prefill — but never beats the Top-1 baseline, and our first reward silently let the policy
*game* the efficiency term (reported FLOPs 0.33, true token-weighted 0.99). (iv) 7.5% of references in our benchmark are
demonstrably wrong and 44% of flagged "hallucinations" are broken traps, which changes conclusions about hallucination.
(v) Even after we *manufacture* expert specialisation (per-domain expert LoRA with a contrastive loss), the right expert beats the wrong one by only 3–6 points, an oracle router gains at most 2.6 points over Top-2, and neither fixed MCDM nor a PPO policy beats Top-2 on the specialised model (Δ ≤ 0.4 points, CIs include 0).
(vi) On a second backbone with natural, much larger expert diversity (DeepSeek-VL2-Tiny, 64 routed experts, top-6), the picture reverses: random expert choice drops accuracy to 0, removing the single best expert costs 8.8 points, and Top-3 (half the FLOPs) is within 0.8 points of Top-6. A PPO Dynamic-K policy lands on or marginally above the Top-K accuracy/FLOPs curve (+0.3 to +0.6 points at matched cost, only one of three CIs excluding 0), but expert usage is only weakly domain-specific. Routing matters when experts are diverse; whether it can be *improved* beyond gate Top-K remains open.
We argue that routing research should report such sensitivity analyses and token-weighted costs.

## 1 Introduction
- Motivation: MoE VLMs (MoE-LLaVA) spend 2 of 4 experts per token; smarter routing (MCDM, RL) promises accuracy/efficiency gains.
- Question: is there routing signal to exploit? Contributions:
  1. **Sensitivity analysis** showing routing-insensitivity of an upcycled VLM-MoE (Table 5) and its weight-space cause.
  2. **DecisionMoE**: TOPSIS + PPO (weights and Dynamic-K) router; 12 ablation variants, 3 seeds for the main one.
  3. **Reward-hacking case study**: step-averaged FLOPs reward is exploited (Fig. 2); token-weighted reward fixes it.
  4. **Benchmark audit**: label-noise and broken-trap analysis changes the hallucination conclusions.
  5. Controlled latency study: where MCDM routing cost comes from (decision matrix 57%, TOPSIS 36%).
  6. **Two-backbone comparison**: on MoE-LLaVA (4 near-interchangeable experts) routing is irrelevant; on DeepSeek-VL2-Tiny (64 diverse experts) it is decisive, Top-3 keeps accuracy at half the FLOPs, and a PPO Dynamic-K policy sits on or marginally above the Top-K frontier (§5.8).

## 2 Related work
Sparse upcycling and MoE VLMs (MoE-LLaVA); routing analyses (expert similarity after upcycling, DeRS); adaptive-K routing
(Ada-K, dynamic expert allocation); RL for routing; MCDM/TOPSIS; hallucination benchmarks (POPE); reward hacking in RL.
*(Tüm 12 künye `references.bib` içinde web ile doğrulandı: upcycling \cite{komatsuzaki2023upcycling}, MoE-LLaVA \cite{lin2024moellava}, DeepSeek-VL2 \cite{wu2024deepseekvl2}, DeRS \cite{huang2025ders}, Ada-K \cite{yue2025adak}, POPE \cite{li2023pope}, PPO/GAE \cite{schulman2017ppo,schulman2016gae}, TOPSIS \cite{hwang1981madm}, reward gaming \cite{skalse2022reward,amodei2016concrete}.)*

## 3 Method
**Setting.** Frozen MoE-LLaVA-Phi2-2.7B×4 (4 experts per MoE layer, nf4-quantised, 8 GB GPU). The router replaces the gate
decision at every MoE layer; one decision (w_t, K_t) per generation step is shared by all layers (design A). Prefill is one step;
each decoded token is one step.

**Criteria.** For each expert: task utility (cosine to a per-layer task prototype), modality match, specialisation
(1 − H(phase | expert)/log 4), reliability, latency, FLOPs share, load balance. Reliability and latency are constant tables
(not measured; `DEVIATIONS.md` C1–C2).

**TOPSIS.** Weighted normalised decision matrix → closeness coefficient; top-K by closeness; selected gate probabilities renormalised.

**PPO policy.** State (dim 78): projected hidden state, modality, entropy, uncertainty, phase, load, budget. Action: Dirichlet
weights over criteria and K∈{1..4}. Reward = λ₁·accuracy − λ₂·FLOPs − λ₃·latency − λ₄·imbalance − λ₅·hallucination,
GAE(γ=1, λ=0.95). **FLOPs is token-weighted** (§5.3). Variants: Top-K, Fixed-MCDM, PPO-fixed-K, PPO-direct, DecisionMoE-B (Dynamic-K).

## 4 Experimental setup
- **Data.** 8,803 questions (COCO, GQA, AI2D, TextVQA images; T1 perception, T2 spatial, T3 multi-hop, T4 hallucination trap,
  T5 OCR+reasoning), generated by Qwen2.5-VL, image-level 70/15/15 split (no image shared across splits). Test 1,321.
- **Metrics.** Strict exact-match accuracy (primary), T4 hallucination rate, K, **token-weighted FLOPs fraction**, latency ratio
  (controlled benchmark, GPU-synchronised, 100 questions × 3 repeats), paired bootstrap 95% CIs (10k resamples).
- **External benchmark.** POPE (random/popular/adversarial, 1,000 each).
- **Seeds.** DecisionMoE-B: seeds 42, 1, 2. Baselines and fixed-MCDM are deterministic.

## 5 Results
### 5.1 Expert selection does not matter (Table 5, validation)
400 val questions: Top-1..4 0.398–0.405; random K=1/2 0.398/0.385; fixed MCDM 0.395; each expert alone 0.380–0.405.
Image is used (0.405 real vs 0.273 blank image). Same-layer expert weights differ by 3–4%.

### 5.2 Main results (Table 1, Fig. 6, Fig. 7)
Top-1 0.397 vs Top-2 0.394 (Δ=+0.2 [−0.6,+1.1] paired); DecisionMoE-B 0.398 (Δ vs Top-1 +0.2 [−0.8,+1.1]) with token-weighted
FLOPs 0.252 vs 0.250 (Top-1) and 0.500 (Top-2). All accuracy differences are inside their 95% CIs. Hallucination: no difference
between variants (raw 0.128–0.143; Table 1).
**Seeds (Table 3).** Accuracy 0.398/0.398/0.396, prefill K = 1.00 in all seeds, token-FLOPs 0.252/0.252/0.253;
only decode K varies by seed (1.40/1.39/1.73), and decode carries little weight in token-weighted cost.
**Fig. 3/4.** K and expert-activation shares are essentially identical across question types: the policy does not specialise
by task, consistent with §5.1 — there is no routing signal to learn.

### 5.3 Reward hacking (Fig. 2)
With a step-averaged FLOPs term, ~600-token prefill counts as one step. The policy chose K=4 in prefill and K≈1 in decode:
reported FLOPs 0.33, but **token-weighted 0.99** (latency 1.21× baseline). Token-weighting the reward (and λ₂: 0.1→1.0)
yields K=1 prefill and FLOPs 0.25. Lesson: efficiency rewards must be weighted by the compute they price.

### 5.4 External validity: POPE (Table 4)
Top-1 0.861, Top-2 0.856, Fixed-MCDM 0.861, PPO-direct (K=2) 0.852; DecisionMoE-B 0.850/0.848/0.846 (seeds 42/1/2; mean K 1.51/1.42/1.50).
Paired differences vs Top-2: DecisionMoE-B −0.67 [−1.23,−0.10], −0.87 [−1.40,−0.37], −1.07 [−1.60,−0.57]; PPO-direct −0.47 [−0.80,−0.17];
Fixed-MCDM +0.43 [−0.03,+0.90]; Top-1 +0.47 [0.00,+0.93]. The three seeds agree: the PPO-trained policies (both Dynamic-K and fixed K=2)
are slightly but significantly *worse* out of distribution than the untrained baselines, while untrained Fixed-MCDM and Top-1 are not.
We read this as mild over-fitting of the policy to our synthetic distribution (no routing signal to learn, §5.1), not as a routing benefit.

### 5.5 Latency (Table 1, profile)
Top-1 0.89×; DecisionMoE-B 1.01×; Fixed-MCDM 1.08×. Fixed-MCDM adds ≈ 8–9 % per question: decision matrix 57%, TOPSIS 36% of
the overhead, tracker CPU↔GPU copy 27% (overlapping). Keeping tracker state on the GPU reduced the overhead from 9.3% to 7.8%.

### 5.6 Benchmark quality
Model-assisted audit (not human; 200 test questions, 40/type): definite wrong reference 7.5% [4.6, 12.0], unverifiable ≈40%,
problematic question 26% (T2: 63%). Of 39 T4 questions that any variant counted as hallucination, 17 are broken traps (object present),
11 ambiguous, 11 genuine: corrected T4 hallucination for Top-2 falls 0.136 → 0.082 (→0.041 excluding ambiguous), with no variant differences.
T3 strict accuracy (0.154) partly reflects metric artefact (token-F1 0.266, MiniLM cos≥0.7 0.294).

### 5.7 Manufacturing specialisation (Table 6, `experiments/results/specialization_summary.json`)
Diagnosis: expert outputs on the same input are *not* identical (mean cosine 0.86, relative L2 0.51 across 16 layers; `expert_diagnosis.json`),
yet accuracy is flat, so the weight-space argument of §5.1 is incomplete: the experts differ, the tasks simply do not reward choosing between them.
We therefore built four distant domains from existing data (diagram / scene text / photo reasoning / photo yes-no; `src/data/domains.py`), assigned one expert per domain,
and trained expert-only LoRA (frozen backbone and gate, train split only) routing domain-d data to expert d; a contrastive variant (L_right + λ·relu(L_right − L_wrong + m)) was added.
Validation (1,074 questions, ≤300 per domain): original model: right-minus-wrong expert −0.2 [−2.2,+1.7] pp; LoRA +4.4 [+2.2,+6.8]; LoRA+contrast +5.5 [+3.5,+7.6];
a stronger contrast (λ=2, m=1) *reduced* it to +3.3 [+1.0,+5.5]. Oracle routing (domain→its expert) beats Top-2 by +1.1 [−1.2,+3.4], +2.6 [+0.5,+4.8], +0.8 [−1.5,+3.3] pp respectively.
Test (1,321, specialised model of the contrast run): Top-2 0.466; fixed MCDM with domain prototypes 0.463 (Δ −0.4 [−1.3,+0.5]); PPO-direct 0.466 (Δ −0.1 [−1.1,+0.9]); no domain shows a difference.
Fine-tuning itself raised accuracy (0.394 → 0.466 on test), but that gain is shared by all experts and is not a routing effect. Caveat: LoRA training used train-split questions from the same distribution as the test set.

### 5.8 A second backbone: DeepSeek-VL2-Tiny (Table 7, Fig. 8; `ds/`, `experiments/results/ds_*.json`)
DeepSeek-VL2-Tiny (3.37B parameters, nf4; 12 MoE layers, 64 routed + 2 shared experts, top-6 gate) is run on the same questions with one (K_t) decision per generation step shared by all layers; FLOPs are token-weighted over routed experts.
**Sensitivity (400 val).** Top-6 0.435; random 6 experts 0.000; bottom-6 0.000; dropping the top-ranked expert (skip-top) 0.348 (−8.8 pp); Top-3 0.425, Top-2 0.423, Top-1 0.373. Unlike §5.1, expert identity matters enormously.
**Pareto (test, 1,321; Table 7).** Top-6 0.471 (FLOPs 1.00), Top-4 0.463 (0.67), Top-3 0.463 (0.50), Top-2 0.443 (0.33), Top-1 0.407 (0.17). Top-3 is −0.8 [−2.0,+0.3] pp vs Top-6 at half the cost.
Dynamic-K policies (PPO, λ=0.05/0.15/0.40): 0.466 @0.667, 0.447 @0.335, 0.413 @0.168. At matched FLOPs vs Top-K: +0.30 [+0.08,+0.61], +0.38 [−0.15,+0.91], +0.60 [−0.53,+1.74] pp.
The policy uses small K in prefill (1–4) and larger K in decode (3–5), which is what token-weighted cost rewards. Honest reading: Dynamic-K matches or marginally exceeds the Top-K frontier; it does not dominate it.
**Natural specialisation (150/domain, `ds_specialization.json`).** Jaccard overlap of each domain's 10 most-used experts: 0.39–0.56 between diagram/scene-text and the others, 0.92 between the two photo domains. Banning a domain's top-10 experts does not hurt that domain more than other domains (e.g. diagram 0.493→0.480; scene-text 0.507→0.507; random-ban control 0.453–0.507; n=150 so ±4 pp is noise): usage is partly domain-specific, function is redundant.
**TOPSIS over 64 experts (K=3, domain given by oracle, `ds/mcdm.py`, `ds_mcdm_{val,test}.json`).** Criteria per layer: gate probability, domain prior (train usage share), specialisation, load. Sanity: gate-only TOPSIS = Top-3 exactly. Val (400): the best of five weightings (0.4/0.3/0.2/0.1) beat Top-3 by +1.7 [+0.25,+3.5] pp, but a heavier prior weight hurt (0.2/0.8: −3.0 [−6.3,+0.3]). Selected on val and confirmed once on test (1,321): 0.462 vs Top-3 0.463 (−0.08 [−0.68,+0.53]; 0.7/0.3 weighting: +0.08 [−0.45,+0.61]). The val gain was selection noise: even with an oracle domain label, MCDM re-ranking does not improve on the gate.
**POPE (3,000 questions, `ds_pope.json`).** Top-6 0.886, Top-3 0.889, Top-2 0.887, dyn_l015 0.887 (FLOPs 0.334, K=2 throughout; identical predictions to Top-2), dyn_l005 0.887. All paired differences vs Top-6 are within ±0.8 pp (e.g. Top-2 +0.13 [−0.53,+0.80]): yes/no object presence needs no more than 2 of 6 experts, so POPE gives no evidence for or against adaptive K. Unlike §5.4, no PPO policy is worse out of distribution.
**Seeds (dyn, λ=0.15, test).** seed 0: 0.447 @0.335 (K_prefill 2); seed 1: 0.415 @0.169 (collapsed to K=1 in prefill); seed 2: 0.445 @0.335. Each lands on the Top-K frontier at its own cost (vs Top-1 +0.75 [−0.30,+1.82]; vs Top-2 +0.23 [−0.15,+0.61]), but the operating point is seed-dependent, so λ does not fix the budget.

## 6 Discussion and limitations
- Why routing is flat: experts are functionally different but the tasks do not reward choosing among them; even manufactured specialisation leaves an oracle ceiling of ≤2.6 points (§5.7). (Earlier draft claim that experts are 'nearly identical' is only true in weight space.) DecisionMoE-style routing would
  need diverse experts (e.g. trained-from-scratch or heavily fine-tuned MoE) — a second backbone (§5.8) confirms that routing matters once experts are diverse.
- Criteria Reliability/Latency were not measured (constant), nf4 quantisation, one external benchmark, model-assisted audit with a
  single labeller (no inter-annotator agreement), audit covers only flagged T4 questions.
- On MoE-LLaVA, Dynamic-K never beats the K=1 baseline; on DeepSeek-VL2-Tiny it is on or marginally above the Top-K frontier (§5.8). We do not claim a large Pareto gain.

## 7 Conclusion
Routing sophistication cannot help when experts are interchangeable, and on a backbone with diverse experts the gate's Top-K is already a strong frontier that a learned Dynamic-K policy only matches or marginally exceeds. Report routing-sensitivity analyses, use token-weighted
costs in rewards and metrics, and audit benchmark labels before drawing hallucination conclusions.

## Reproducibility
`README.md` (full pipeline), `DEVIATIONS.md` (deviations/limitations), per-result `meta` provenance; code and result JSONs in repo.
