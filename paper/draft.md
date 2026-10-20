# When Does Expert Routing Matter? Multi-Criteria and Reinforcement-Learned Routing in Two Vision-Language MoE Models

*Taslak (iskelet + gerçek sayılar). Tablolar `paper/tables/t1–t5.tex`, figürler `paper/figures/f2–f7`. Sayılar test setinden
(1321 soru) ve `experiments/results/` dosyalarındandır; işaretli yerler ek koşular bitince güncellenir.*

## Abstract
Sparse Mixture-of-Experts (MoE) vision-language models route each token to a few experts, and many proposals promise smarter routers. We test the premise that routing carries exploitable signal, on two frozen backbones and one 8 GB GPU, with DecisionMoE: a router that ranks experts with TOPSIS and lets a PPO policy choose criterion weights and the number of active experts K per step, compared with the gate's Top-K at matched, token-weighted expert FLOPs.
(i) On a sparsely upcycled model (MoE-LLaVA-Phi2-2.7B×4, 8,803 synthetic questions, POPE) accuracy is insensitive to expert selection: Top-K for K=1..4, random routing, single experts and fixed MCDM routing all fall in a 2.5-point band, and a learned policy only rediscovers K=1.
(ii) On a model with 64 natively trained experts (DeepSeek-VL2-Tiny) the picture reverses: random experts give 0.000 accuracy and dropping the best expert costs 8.8 points, yet Top-3 is within 0.8 points of Top-6 at half the FLOPs. TOPSIS re-ranking adds nothing, even with an oracle domain label.
(iii) Adaptive K helps only with long generation. With brief chain-of-thought on AI2D and ScienceQA-IMG, retrained PPO policies beat the matched-FLOPs Top-K mixture by about 1–3 points at λ=0.4 (positive in all six seeds, CIs excluding zero in four), but the gain is task-specific (AI2D policies do not transfer), unstable at stronger cost pressure, and a gate-mass heuristic matches it at one setting.
(iv) A step-averaged FLOPs reward is gamed by the policy (reported 0.33, true token-weighted 0.99), and 7.5% of benchmark references are wrong, with 44% of flagged hallucinations being broken traps.
Routing matters when experts are diverse, but beyond the gate's Top-K the achievable gain is small; we recommend reporting sensitivity analyses and token-weighted costs.

## 1 Introduction
Sparse Mixture-of-Experts (MoE) vision-language models activate only a few experts per token, and a growing literature proposes smarter routers: multi-criteria decision making (MCDM), reinforcement learning, and adaptive numbers of active experts. Each proposal presupposes that there is routing signal to exploit, i.e. that *which* experts, and *how many*, changes the answer. We test that premise before building on it.

We study two frozen backbones on one 8 GB GPU: MoE-LLaVA-Phi2-2.7B×4 (a sparsely upcycled model with four near-interchangeable experts) and DeepSeek-VL2-Tiny (64 natively trained routed experts, top-6). On both we implement DecisionMoE, a router that ranks experts with TOPSIS and lets a PPO policy choose criterion weights and a per-step number of active experts K, and we compare it with the gate's own Top-K at matched, token-weighted expert FLOPs.

Contributions.
1. **Sensitivity analysis on two backbones.** On the upcycled model accuracy is insensitive to expert selection (Table 5) and the cause is visible in weight space; on DeepSeek-VL2-Tiny expert identity matters enormously (random experts give 0.000 accuracy), yet Top-3 keeps accuracy at half the FLOPs (§5.8).
2. **A router that does not beat the gate, and why.** DecisionMoE (TOPSIS + PPO) never beats Top-K on short answers; TOPSIS re-ranking adds nothing even with an oracle domain label, because its informative criteria are what the gate already encodes (§5.2, §5.8).
3. **When adaptive K does help.** Only with long generation (brief chain-of-thought on AI2D and ScienceQA-IMG), where a retrained PPO policy beats the matched-FLOPs Top-K mixture by about 1–3 points, with seed- and task-dependence and without a demonstrated edge over a gate-mass heuristic (§5.9).
4. **Reward-hacking case study.** A step-averaged FLOPs reward is exploited (reported 0.33, true 0.99); a token-weighted reward fixes it (§5.3).
5. **Benchmark audit.** 7.5% of references in our synthetic benchmark are wrong and 44% of flagged hallucinations are broken traps (§5.6).
6. Negative pilots (expert pools for IO savings, token-level top-p K in the short-answer regime) reported rather than discarded (§5.9).

## 2 Related work
**Sparse upcycling and MoE VLMs.** Sparse upcycling \cite{komatsuzaki2023upcycling} initialises every expert from one dense feed-forward block; MoE-LLaVA \cite{lin2024moellava} applies it to a vision-language model. DeRS \cite{huang2025ders} and related analyses show that upcycled experts remain close in weight space, which is consistent with our finding that their selection is irrelevant. DeepSeek-VL2 \cite{wu2024deepseekvl2} instead trains fine-grained experts natively.
**Adaptive and learned routing.** Ada-K \cite{yue2025adak} and related work vary the number of active experts per token, in some cases with reinforcement learning. We differ by comparing against the gate's Top-K at matched token-weighted cost on a frozen model and by reporting where such policies fail.
**MCDM.** TOPSIS \cite{hwang1981madm} ranks alternatives by closeness to an ideal solution over weighted criteria; we use it as an expert ranker.
**Evaluation pitfalls.** POPE \cite{li2023pope} measures object hallucination; reward misspecification and gaming \cite{skalse2022reward,amodei2016concrete} motivate our reward-hacking analysis; the policy is trained with PPO and GAE \cite{schulman2017ppo,schulman2016gae}.

## 3 Method
**Routing interface.** Both backbones are frozen and nf4-quantised. A router replaces the gate decision at every MoE layer. One decision per generation step (prefill is one step, each decoded token another) is shared by all layers. The cost measure is the **token-weighted fraction of routed-expert FLOPs**, K/K_max averaged over tokens, so the ≈600-token prefill dominates (this is what the step-averaged reward got wrong, §5.3).

**TOPSIS expert ranking.** For each candidate expert a decision matrix of criteria is built, normalised, weighted and scored by closeness to the ideal solution; the top-K by closeness are selected and their gate probabilities renormalised. On MoE-LLaVA (4 experts) the seven criteria are task utility, modality match, specialisation, reliability, latency, FLOPs share and load (reliability and latency are constant tables; `DEVIATIONS.md` C1–C2). On DeepSeek-VL2-Tiny (64 experts) the criteria are gate probability, a domain prior (train usage share, given by an oracle domain label), specialisation and load.

**PPO policy.** On MoE-LLaVA the state (dim 78) comprises a projected hidden state, modality, entropy, uncertainty, phase, load and budget; the action is a Dirichlet distribution over criterion weights plus K∈{1..4}. On DeepSeek-VL2-Tiny (`DynKPolicy`) the action is K∈{1..6} per step. Reward = accuracy − λ·(token-weighted FLOPs fraction) (MoE-LLaVA adds latency, imbalance and hallucination terms), GAE(γ=1, λ=0.95), clipped PPO. Variants: Top-K, Fixed-MCDM, PPO-fixed-K, PPO-direct, DecisionMoE-B (Dynamic-K).

**Heuristic baseline.** "Top-p": per token and layer, the smallest j such that the top-6 gate mass reaches τ. It has no learned parameters and tests whether a policy is needed at all.

## 4 Experimental setup
- **Synthetic benchmark (both backbones).** 8,803 questions (COCO, GQA, AI2D, TextVQA images; T1 perception, T2 spatial, T3 multi-hop, T4 hallucination trap, T5 OCR+reasoning), generated by Qwen2.5-VL, image-level 70/15/15 split. Test 1,321. Answers are short.
- **Reasoning benchmarks (DeepSeek-VL2-Tiny).** AI2D multiple choice (3,088 questions, split by image hash: train 2,150 / val 341 / test 597) and ScienceQA-IMG (2–4 options; same sizes; official splits subsampled with a fixed seed). A brief chain-of-thought prompt ("Think step by step briefly, then finish with 'Answer: <letter>'", ≤128 new tokens); the last stated letter is scored.
- **External benchmark.** POPE (random/popular/adversarial, 1,000 each).
- **Metrics.** Exact-match accuracy, K, token-weighted FLOPs fraction, latency ratio (GPU-synchronised, 100 questions × 3 repeats), and paired bootstrap 95% CIs (10k resamples). The matched-FLOPs baseline is the linear mixture of the two adjacent Top-K points; where the Top-K frontier is non-convex (ScienceQA) we also compare with the real Top-K points.
- **Seeds and selection.** DecisionMoE-B on MoE-LLaVA: seeds 42, 1, 2. PPO on DeepSeek-VL2-Tiny: 80 updates × 48 episodes, seeds 0–2 at λ=0.4. Hyper-parameters are chosen on validation and evaluated once on test. We report all runs, not the best seed, and do not correct for multiple comparisons (§6).
- **Hardware and reproducibility.** One 8 GB RTX 5060; training and evaluation scripts resume from checkpoints after interruption.

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

### 5.9 When does adaptive K help? Pilots and the long-generation regime (`ds/pool_pilot.py`, `ds/topp_pilot.py`, `ds/cot_pilot.py`, `experiments/results/ds_*pilot.json`, `ds_eval_ai2d_test.json`)
Three pilots on DeepSeek-VL2-Tiny (val, 400 questions) motivated where adaptive routing can matter at all.
**Expert pool (IO/cache motivation).** Top-6 activates all 64 experts per layer over a ≈600-token prefill, so we restricted each layer to a pool of the m most-used experts (prefill gate mass) and routed top-K inside it. Accuracy collapses: m=32 0.320, 24 0.255, 16 0.110, 8 0.013 (Top-6 0.435; random pool m=32 0.340, m=16 0.048). A smaller expert footprint is not available through pool restriction, so an IO-bound formulation was abandoned (no-go).
**Token-level adaptive K, short answers.** Choosing per token the smallest j such that the top-6 gate mass reaches τ ("top-p", a hand-made heuristic) gives 0.422 @0.386, 0.427 @0.448, 0.430 @0.567–0.773 FLOPs (τ=0.4–0.8), i.e. on the Top-K frontier (Top-4 0.435 @0.667) but not above it: with ≈2 decoded tokens, per-step K only controls the prefill and the gate is already near-optimal.
**Long generation.** With a brief chain-of-thought (≤128 new tokens, mean ≈32) on AI2D multiple choice (200 questions) accuracy depends steeply on K: Top-6 0.53, Top-3 0.47, Top-2 0.41, Top-1 0.24, and lowering K only at decode costs up to 8 pp (K_decode=1: 0.45), so the generation phase is where routing can be non-flat.
**AI2D reasoning regime (test, 597 questions; images split by hash, train 2,150 / val 341 / test 597).** Top-K frontier: K=6 0.633 (FLOPs 1.00), 5 0.615 (0.83), 4 0.575 (0.67), 3 0.518 (0.50), 2 0.461 (0.33), 1 0.313 (0.17). Gain over the matched-FLOPs mixture of adjacent Top-K points (paired bootstrap, 95% CI):

| Policy | Acc. | FLOPs | Gain (pp) |
|---|---|---|---|
| PPO λ=0.4, seed 0/1/2 | 0.486/0.484/0.476 | 0.344/0.344/0.338 | +2.14 [0.33,4.02] / +2.00 [0.15,3.86] / +1.34 [−0.43,3.20] |
| PPO λ=0.8, seed 0/1/2 | 0.369/0.320/0.322 | 0.199/0.171/0.171 | +2.63 [0.57,4.74] / +0.33 [−1.07,1.80] / +0.47 [−0.87,1.82] |
| PPO λ=0.2, seed 0 | 0.462 | 0.338 | +0.02 [−1.70,1.75] |
| top-p τ=0.3/0.4/0.5/0.6/0.7 | 0.394/0.461/0.533/0.528/0.566 | 0.295/0.395/0.449/0.571/0.683 | −3.34 / −2.12 / +3.24 [0.25,6.15] / −1.43 / −1.22 |

Honest reading: a learned token-/layer-level K policy beats the matched-FLOPs Top-K mixture by ≈1.3–2.1 pp at λ=0.4 (sign positive in all three seeds, CI excludes zero in two), but the gain is small, λ=0.8 is seed-unstable (seeds 1–2 collapse to K=1 in prefill and gain ≈0), and the heuristic wins at one τ out of five (+3.2) while losing at the others, so we cannot claim PPO beats the heuristic. TOPSIS/MCDM never contributed (§5.8). Caveats: one backbone, AI2D may overlap DeepSeek-VL2's training data, several policies were compared on one test set (no multiplicity correction).
**ScienceQA-IMG replication (test, 597 questions; `ds_eval_sqa_test.json`, Table 8, Fig. 9).** Same protocol, 2–4-option questions, hint prepended. The Top-K frontier differs from AI2D: K=6 0.792, 5 0.817, 4 0.755, 3 0.755, 2 0.568, 1 0.385 (FLOPs 1.00/0.83/0.67/0.50/0.33/0.17), so there is a cliff between K=2 and K=3 and K=3–4 are free. *Transfer* of the AI2D-trained policies fails: dyn_a040 seeds 0/1/2 are −3.2 [−5.4,−1.1], −2.5 [−4.7,−0.2], −2.6 [−4.9,−0.3] pp below the matched-FLOPs Top-K mixture; the τ heuristic is −9.2 [−12.0,−6.3] (τ=0.4), −0.2 [−3.3,+2.8] (τ=0.5), +0.5 [−1.7,+2.7] (τ=0.6). Policies *retrained* on ScienceQA (λ=0.4, 80 updates × 48 episodes, 3 seeds) beat the mixture in all three seeds: +3.5 [+1.6,+5.5] (0.786 @0.495), +1.0 [−0.7,+2.7] (0.765 @0.502), +8.2 [+5.7,+10.9] (0.727 @0.401). Read against the real Top-K points rather than the mixture: seeds 0 and 1 match or slightly exceed Top-3 (0.755 @0.500; +3.1 and +1.0 pp) and fall 0.6–3.9 pp short of Top-5/6 at 40–50% lower cost; seed 2 (0.727 @0.401) trades 2.8 pp against Top-3 for 20% fewer FLOPs. Because the frontier is non-convex here the linear mixture is a weak baseline and the +8.2 overstates the gain; the robust statement is "retrained adaptive K lands at or above the Top-3 operating point and is on the frontier elsewhere". The learned policy is task-specific (it must be retrained per task), and the heuristic does not give a consistent gain on either dataset.

## 6 Discussion and limitations
**Why routing is flat on one backbone and sharp on the other.** In the upcycled model the experts differ by only 3–4% in weights, so selection is irrelevant; even manufactured specialisation leaves an oracle ceiling of ≤2.6 points (§5.7). Natively trained experts are functionally different (§5.8), yet the gate already picks near-optimally for short answers, and giving TOPSIS the criteria the gate knows (probability, domain prior) cannot add information. MCDM could only help with criteria the gate cannot see (cost, memory, cache state); our expert-pool pilot (§5.9) shows that restricting the pool to realise such savings destroys accuracy.
**What adaptive K buys.** Only in long generation does the K-accuracy curve become steep enough for a per-step policy to exploit. There the gain is 1–3 points at matched cost, depends on the seed and on λ, requires retraining per task, and a hand-made gate-mass heuristic matches it at one setting; we therefore do not claim that PPO is necessary, nor a large Pareto gain.
**Limitations.**
- Two backbones, both small and nf4-quantised; no larger MoE-VLM fits an 8 GB GPU. Cost is modelled as token-weighted routed-expert FLOPs, not measured latency or memory traffic.
- AI2D and ScienceQA-IMG may overlap DeepSeek-VL2's training data; several policies are compared on one test set per dataset without multiplicity correction; the matched-FLOPs mixture is a weak baseline when the Top-K frontier is non-convex (ScienceQA).
- On MoE-LLaVA the reliability and latency criteria are constant tables (`DEVIATIONS.md` C1–C2). The synthetic benchmark was generated by a model and audited by a single labeller on flagged T4 questions only (no inter-annotator agreement); POPE is the only external benchmark on that backbone.

## 7 Conclusion
Routing sophistication cannot help when experts are interchangeable, and on a backbone with diverse experts the gate's Top-K is already a strong frontier that a learned Dynamic-K policy only matches or marginally exceeds it on short answers; on brief chain-of-thought reasoning (AI2D, ScienceQA-IMG) it gains about 1–3 points at matched cost, but only after per-task retraining and without a demonstrated advantage over a gate-mass heuristic. Report routing-sensitivity analyses, use token-weighted
costs in rewards and metrics, and audit benchmark labels before drawing hallucination conclusions.

## Reproducibility
`README.md` (full pipeline), `DEVIATIONS.md` (deviations/limitations), per-result `meta` provenance; code and result JSONs in repo.
