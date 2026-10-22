# DecisionMoE

**MoE tabanlı görsel-dil modellerinde çok kriterli karar verme (TOPSIS) ve pekiştirmeli öğrenme (PPO) ile expert yönlendirme**

`Python 3.12` · `PyTorch` · `MoE-LLaVA-Phi2-2.7B×4` · `DeepSeek-VL2-Tiny` · `TOPSIS + PPO` · `Dynamic-K`

**İçindekiler:** [Bulgular](#1-ne-yapıyor-ne-buldu) · [Depo yapısı](#2-depo-yapısı) · [Kurulum](#3-kurulum) · [Süreç](#4-baştan-sona-süreç) · [Testler](#5-testler) · [Tekrarlanabilirlik](#6-tekrarlanabilirlik-notları)

**Hızlı bakış**

| | |
|---|---|
| Soru | Expert seçimi doğruluğu gerçekten etkiliyor mu? |
| Yöntem | 5 kriter → TOPSIS skoru; ağırlık ve K (1–4) PPO politikasından |
| Veri | 8803 soru, 5 soru tipi, COCO/AI2D/GQA/TextVQA + POPE doğrulaması |
| Sonuç | MoE-LLaVA expert seçimine duyarsız (K=1 ≈ K=2). DeepSeek-VL2-Tiny'de seçim çok önemli; uyarlanır K yalnızca uzun üretimli çoktan seçmeli akıl yürütmede (AI2D) Top-K sınırını geçiyor (+2–3 puan, ön bulgu) |
| Durum (19 Ekim) | Deneyler sürüyor (ek seed'ler); TOPSIS/MCDM bileşeni DeepSeek'te ek fayda sağlamadı; ayrıntı aşağıda ve `DEVIATIONS.md` D10–D13 |
| Makale | `paper/` (taslak, tablolar, şekiller) |

> **Not:** Eğitilmiş expert LoRA ağırlıkları (`artifacts/experts_*/lora.pt`, ~100 MB/adet) GitHub sınırı nedeniyle depoda yoktur;
> `scripts/train_experts.py` ile yeniden üretilir. Politika ve prototip dosyaları depodadır.

MoE tabanlı görsel-dil modellerinde (MoE-LLaVA-Phi2-2.7B×4) expert routing'ini **çok kriterli karar verme (TOPSIS)** ve
**pekiştirmeli öğrenme (PPO)** ile yöneten bir araştırma hattı. Hedef: IJCV özel sayısı (son tarih 1 Kasım 2026).

> **Önce okuyun:** `DEVIATIONS.md` (başlangıç tasarımından sapmalar, bilinen sınırlamalar, denetim bulguları).

## 1. Ne yapıyor, ne buldu?

Her MoE katmanında, her token için "hangi expert(ler) çalışsın?" kararı standart olarak router logit'lerinin Top-K'sıyla
verilir. DecisionMoE bu kararı şöyle yeniden kurar:

1. Her expert için kriterler hesaplanır: **Utility** (görev prototipine kosinüs benzerliği), **Specialization**, **Load**,
   **Reliability**, **Latency**.
2. Kriterler **TOPSIS** ile tek bir skora indirilir; kriter ağırlıkları bir **PPO politikasından** (Dirichlet) gelir.
3. Aynı politika **K'yı da** (1–4 expert) seçebilir (Dynamic-K). Ödül = doğruluk − FLOPs − gecikme − yük dengesizliği −
   hallucination.

Ana bulgular (test seti, 1321 soru, tek seed; ayrıntı ve güven aralıkları `experiments/results/`):

- **Bu backbone expert seçimine duyarsız.** Top-K (K=1..4), rastgele seçim, her expert tek başına ve sabit-MCDM hepsi
  0.380–0.405 aralığında. Sebep: model sparse upcycling ile kurulmuş, aynı katmandaki expert ağırlıkları yalnızca ~%3–4
  farklı. Model görüntüyü kullanıyor (gerçek görüntü 0.405, boş görüntü 0.273).
- **K=1, K=2 ile eşit doğrulukta FLOPs'un yarısı** (0.397 vs 0.394, farklar eşleştirilmiş %95 GA içinde sıfırı kapsıyor;
  POPE'ta da aynı: 0.861 vs 0.856).
- **Reward tasarımı kritik:** adım bazlı FLOPs ödülünde politika prefill'de (~600 token) K=4, decode'da K≈1 seçip
  "FLOPs 0.33" gösterdi ama gerçek token-ağırlıklı FLOPs ≈ 0.99'du (**reward hacking**). Token-ağırlıklı FLOPs ve λ₂=1.0 ile
  politika K≈1.4'e (FLOPs 0.25) iniyor: RL, K=1'in yeterli olduğunu kendiliğinden buluyor ama K=1 baseline'ını geçmiyor.
- **Etiket kalitesi sınırlı:** 200 soruluk model-destekli denetimde kesin yanlış referans %7.5, sorunlu soru %26;
  hallucination (T4) tuzaklarının bir kısmı bozuk. Düzeltilmiş hallucination oranları `analyze_results.py` çıktısında.
- **MCDM'nin maliyeti:** sabit-MCDM gecikmesi top-2'ye göre ~%8–9 (karar matrisi %57, TOPSIS %36 — profil
  `profile_router.py`). Dynamic-K bunu K düşüşüyle geri ödüyor (1.007×) ama K=1 baseline'ı (0.89×) daha hızlı.

Makalenin konumu: **analiz sonucu** — MoE-LLaVA'da routing'in doğruluğa etkisi yok; DeepSeek-VL2-Tiny'de etkisi büyük ama gate Top-K kısa cevaplı görevde zaten yakın-optimal; uyarlanır K ancak uzun üretimli akıl yürütmede kazandırıyor. Ödül tasarımı kritik.

## 2. Depo yapısı

```
configs/                 model.yaml, ppo.yaml (ödül ağırlıkları), topsis.yaml
src/backbone/            loader.py (nf4 yükleme), prototype.py (görev prototipleri)
src/routing/             criteria.py (kriter takibi), topsis.py, phase.py, router.py (DecisionMoERouter), runner.py
src/rl/                  policy.py (Dirichlet politika), ppo.py, reward.py
src/data/                dataset.py, collator.py, pope.py
src/utils/               metrics.py, secondary_metrics.py, logger.py, experiment.py, provenance.py
experiments/ablation/    variant_*.yaml (12 konfigürasyon)
experiments/results/     eval_*.json, pope_*.json, latency_benchmark*.json, sensitivity_val.json, denetim dosyaları
artifacts/               prototypes.pt, policy_<variant>[_s<seed>].pt
decisionmoe_dataset/     questions.json, splits/ (görüntüler repoda değil)
pope_data/               POPE soru dosyaları (görüntüler repoda değil)
scripts/                 çalıştırılabilir betikler (aşağıda)
tests/                   pytest
paper/                   style.py, figures/, tables/
third_party/             MoE-LLaVA (setup_moe_env.sh klonlar; repoda değil)
```

## 3. Kurulum

İki ortam gerekir (bitsandbytes/transformers sürümleri nedeniyle ayrı tutulur):

- `venv` — veri üretimi ve saf-torch testler: `python -m venv venv && venv/bin/pip install -r requirements.txt`
- `venv_moe` — backbone, router, eğitim: `bash scripts/setup_moe_env.sh`
  (cu128 torch, transformers 4.37.0, accelerate 0.26.1, `moellava --no-deps`, bitsandbytes yaması dahil)

Donanım: RTX 5060 **8 GB**. Backbone **nf4** kuantize yüklenir (int8, bitsandbytes 0.50 + torch 2.11 ile bozuk; bkz.
`DEVIATIONS.md`). Backbone indirme (~13 GB): `bash scripts/start_download.sh`.

## 4. Baştan sona süreç

Her adımın girdi/çıktısı. `[venv]` / `[moe]` hangi ortamda koşulacağını gösterir.

### 4.1 Veri seti üretimi `[venv]`
Hedef: COCO / AI2D / GQA / TextVQA görüntüleri üzerinde 5 soru tipi (T1 algı, T2 uzamsal, T3 çok adımlı, T4 hallucination
tuzağı, T5 OCR+akıl yürütme). Toplam **8803 soru**, 70/15/15 stratified bölünme.

1. Görüntü seçimi: `select_coco.py`, `select_ai2d.py`, `select_gqa_ids.py` + `download_gqa_selected.py`,
   `select_download_textgqa.py`
2. Soru üretimi (LLM): `generate_questions.py` / `run_generate.sh` → `decisionmoe_dataset/raw/`
3. Süzme ve kalite kuralları: `filter_questions.py`, `qc_rules.py` → `questions.json`
4. Bölme: `make_splits.py` → `splits/{train,val,test}.json` (görüntü düzeyinde; train∩val∩test = ∅ doğrulandı)
5. Doğrulama: `validate_dataset.py`

Görüntüler (~420 MB) repoda yoktur; kaynak veri setlerinden yukarıdaki betiklerle yeniden indirilir.
`questions.json` ve `splits/` repoda vardır.

### 4.2 Backbone ve prototipler `[moe]`
- `python scripts/smoke_backbone.py` — backbone yükleniyor mu, generate çalışıyor mu
- `python scripts/compute_prototypes.py` → `artifacts/prototypes.pt` (train split'ten görev başına prototip; utility kriterinin girdisi)
- `python scripts/smoke_router.py` — router hook'ları

### 4.3 Duyarlılık testi `[moe]`
`python scripts/sensitivity_test.py 400` → `experiments/results/sensitivity_val.json`. 400 val sorusunda Top-K, rastgele
ve tek-expert yönlendirmelerini karşılaştırır. **Bu, makalenin ana bulgusunu veren deneydir** (backbone expert seçimine duyarsız).

### 4.4 Eğitim `[moe]`
```bash
python scripts/train.py --variant decisionmoe_b --updates 150 --resume --no-wandb [--seed N]
```
PPO yalnızca politikayı eğitir (backbone donuk). Politika → `artifacts/policy_<variant>.pt` (seed ≠ 42 ise `_s<seed>`).
Ödül: `src/rl/reward.py`; FLOPs terimi **token-ağırlıklı** (`src/rl/ppo.py::episode_rewards`). Eğitim sırasında her 25/50
güncellemede val üzerinde doğruluk/K/FLOPs raporlanır.

Eğitim gerektirmeyen variant'lar (`standard_topk`, `standard_topk_k1`, `fixed_mcdm_*`) doğrudan değerlendirilir.
Variant tanımları: `experiments/ablation/variant_*.yaml`.

| Variant | Açıklama |
|---|---|
| `standard_topk`, `standard_topk_k1` | Orijinal router, K=2 / K=1 (baseline) |
| `fixed_mcdm_{no_utility,cosine}` | Sabit ağırlıklı TOPSIS, K=2 |
| `ppo_fixedk_{no_utility,cosine}`, `ppo_direct_fixedk` | PPO ağırlıkları, K=2 sabit |
| `ppo_direct_dynk`, `decisionmoe_{a,b}` | PPO + Dynamic-K (b: kosinüs proxy, **ana yöntem**) |
| `decisionmoe_b_{l03,l30}` | λ₂ taraması (kullanılmadı) |

### 4.5 Değerlendirme `[moe]`
```bash
python scripts/evaluate.py --variant decisionmoe_b [--split test] [--seed N] [--limit 100]
```
Çıktı `experiments/results/eval_<variant>_test.json`: doğruluk, hallucination, K, FLOPs (token-ağırlıklı), soru bazlı
`items` (tahmin, doğru mu, K dizisi, expert kullanımı, kriter ağırlıkları) ve `meta` (provenance: commit, config özeti,
paket sürümleri, seed, GPU). Aynı kod yolu `--limit 100` ile eski ve yeni sürümde birebir aynı tahminleri üretir.

> Dikkat: `evaluate.py` mevcut `eval_*.json` dosyasının üzerine yazar. `--limit` ile yapılan denemelerden sonra dosyayı geri yükleyin.

### 4.6 Dış doğrulama: POPE `[moe]`
```bash
python scripts/prepare_pope.py                       # pope_data/ (COCO tabanlı evet/hayır nesne varlığı)
python scripts/evaluate_pope.py --variants standard_topk standard_topk_k1 decisionmoe_b   # → pope_<variant>.json
```
Doğruluk, F1, "yes" oranı; random/popular/adversarial ayrı ayrı (1000'er soru).

### 4.7 Gecikme ve profil `[moe]`
```bash
python scripts/benchmark_latency.py --variants standard_topk standard_topk_k1 fixed_mcdm_cosine ppo_direct_fixedk decisionmoe_b
python scripts/profile_router.py
```
Kontrollü gecikme (ısınma + 3 tekrar, GPU senkronlu) → `latency_benchmark.json`. Tabloya girecek gecikme **buradan**
alınır; koşu içi duvar saati kullanılmaz. `latency_benchmark_pre_opt.json`: tracker'ı GPU'ya almadan önceki ölçüm.
`profile_router.py` router gecikmesini karar matrisi / TOPSIS / tracker kopyası olarak ayrıştırır.

### 4.8 Ölçüm ve etiket denetimi `[venv]`
- `python scripts/rescore.py` — ikincil metrikler (token-F1, MiniLM kosinüs eşiği) → `rescored.json`
  (T3 referansları uzun cümle, model kısa cevap veriyor: katı eşleşme T3'ü olduğundan düşük gösterir)
- `python scripts/make_audit_sheet.py` → `audit_sheet.csv` (200 test sorusu, tip başına 40); `analyze_audit.py` özetler.
  Etiketleme **model-destekli** (insan değil); kesin/belirsiz ayrımına dikkat (`DEVIATIONS.md`).
- T4 hallucination düzeltmesi: `t4_audit.json` (bozuk tuzak sayısı); `analyze_results.py` düzeltilmiş oranı hesaplar.

### 4.9 Analiz `[venv]`
```bash
python scripts/analyze_results.py
```
Tablo: variant başına doğruluk, top-2 ve top-1'e göre eşleştirilmiş fark + bootstrap %95 GA, hallucination (ham/düzeltilmiş),
K, FLOPs, gecikme; Pareto baskınlık notu; `seed_summary.json` varsa seed ortalaması ± std.
Çıktılar: `ablation_scores.json`, `efficiency_metrics.json`, `expert_utilization.json`, `criteria_weights.json`.

### 4.10 Makale malzemesi `[venv]`
```bash
python scripts/analyze_pope.py                  # POPE eşleştirilmiş farklar → pope_summary.json
python scripts/generate_paper_materials.py      # paper/tables/t1–t5.tex, paper/figures/f2–f7 (.pdf/.png)
```
`paper/draft.md`: makale taslağı (gerçek sayılarla). Ek seed işleri: `scripts/run_seeds.sh` (decisionmoe_b seed 1,2: eğitim + test),
`scripts/run_extra.sh` (ek POPE koşuları; `evaluate_pope.py --policy-seed`).

### 4.11 Takip
`python3 scripts/progress.py --watch` — çubuklu ilerleme raporu. Uzun işler (`scripts/run_priority.sh`, `run_seeds.sh`)
**VSCode terminalinden değil**, `setsid nohup bash scripts/<betik>.sh > <log> 2>&1 < /dev/null &` ile başlatılmalıdır
(VSCode çökerse süreç kilitlenir). `scripts/start_priority.sh` bunun sarmalayıcısıdır.

## 5. Testler

```bash
venv/bin/python -m pytest -q tests          # saf-torch testler (DeepSpeed gerektirenler atlanır)
venv_moe/bin/python -m pytest -q tests      # tümü (router, prototype, tracker GPU eşdeğerliği)
ruff check src scripts tests paper
```

## 6. Tekrarlanabilirlik notları

- Sonuç JSON'larındaki `meta` alanı: git commit, config özetleri, paket sürümleri, seed, GPU.
- Seed 42 eski adlandırmayı korur; diğer seed'ler `_s<seed>` eki alır (`policy_decisionmoe_b_s1.pt`, `eval_decisionmoe_b_s1_test.json`).
- Eski/denenmiş koşular `experiments/results/archive_*` ve `artifacts/archive_*` altında (repoya alınmaz):
  `archive_run1` (ilk koşu), `archive_run2_stepflops` (reward hacking kanıtı).
- Eski betikler `scripts/legacy/` altında (kullanılmıyor).
- Bilinen sınırlamalar: tek backbone, nf4 kuantizasyon, Reliability/Latency kriterleri ölçülmedi (nötr tablo), etiket denetimi
  model-destekli, standart dış benchmark yalnızca POPE.

## Uzmanlaşma deneyi (ek)
`scripts/diagnose_experts.py` (expert çıktı benzerliği) → `scripts/train_experts.py` (alan başına expert LoRA, isteğe bağlı kontrast) →
`scripts/domain_sensitivity.py` (alan × expert doğruluk matrisi) → `scripts/compute_domain_prototypes.py` + `scripts/run_spec.sh` (Top-2, fixed MCDM, PPO-direct; `DM_TAG=spec`) →
`scripts/analyze_specialization.py` (özet + `paper/tables/t6_specialization.tex`). Sonuç: bkz. `paper/draft.md` §5.6.

## İkinci backbone: DeepSeek-VL2-Tiny (`ds/`, venv_ds)
64 doğal uzman expert + 2 paylaşımlı, top-6, 12 katman (3.37B, nf4). Tüm betikler `venv_ds` ile; görev `DS_TASK=synth|ai2d` ortam değişkeniyle seçilir (`ds/task.py`).

| Adım | Betik | Çıktı (`experiments/results/`) |
|---|---|---|
| Duyarlılık (val 400) | `ds/sensitivity.py` | `ds_sensitivity_val.json`: rastgele→0, en iyi expert çıkınca −8.8 puan |
| Dynamic-K PPO + test (sentetik, kısa cevap) | `scripts/run_ds.sh`, `ds/train_dynk.py`, `ds/eval_test.py` | `ds_eval_test.json`; `scripts/analyze_ds.py` → `ds_pareto_summary.json`, `paper/tables/t7_ds_pareto.tex`, `paper/figures/f8_ds_pareto.*` |
| POPE + ek seed | `scripts/run_ds2.sh`, `ds/pope_ds.py` | `ds_pope.json`, `ds_eval_test.json` (l015s1/s2) |
| Doğal uzmanlaşma | `ds/specialization.py` | `ds_specialization.json`: kullanım kısmen alan-spesifik, ablasyon alan-spesifik değil |
| TOPSIS 64 expert (oracle alan) | `ds/mcdm.py`, `ds/mcdm_test.py` | `ds_mcdm_{val,test}.json`: test'te Top-3'ten fark yok |
| Pilot: expert havuzu (IO) | `ds/pool_pilot.py` | `ds_pool_pilot.json`: havuz küçülünce doğruluk çöküyor (no-go) |
| Pilot: token-düzeyi uyarlanır K sezgiseli | `ds/topp_pilot.py` | `ds_topp_pilot.json`: kısa cevap rejiminde Top-K sınırını geçmiyor |
| Pilot: uzun üretim rejimi | `ds/cot_pilot.py`, `scripts/prepare_ai2d.py` | `ds_cot_pilot.json`: bu rejimde K belirleyici |
| AI2D akıl yürütme: PPO + test (597 soru) | `scripts/run_ds3.sh` (`DS_TASK=ai2d`) | `ds_eval_ai2d_{val,test}.json` |
| AI2D pekiştirme (sezgisel τ taraması, ek seed'ler) | `scripts/run_ds4.sh` | `ds_eval_ai2d_test.json` |
| ScienceQA-IMG doğrulaması (transfer + yeniden eğitim, 3 seed) | `scripts/prepare_sqa.py`, `scripts/run_sqa.sh` (`DS_TASK=sqa`, checkpoint'li/yeniden başlatılabilir) | `ds_eval_sqa_test.json`, `scripts/analyze_mcq.py` → `ds_mcq_summary.json`, t8, f9 |

AI2D verisi `scripts/prepare_ai2d.py` ile indirilir (`ai2d_data/`, depoda yok; bölme görüntü özetine göre: train 2150 / val 341 / test 597).

**Bulgular (ayrıntı: `paper/draft.md` §5.2–5.4):** kısa cevaplı sentetik sette Top-3 ≈ Top-6 yarı FLOPs ile ve Dynamic-K Top-K sınırının en fazla +0.3–0.6 puan üstünde (çoğu GA sıfırı kapsıyor). Uzun üretimli AI2D'de sınır eğimli (Top-6 0.633, Top-3 0.518, Top-2 0.461, Top-1 0.313) ve eşleşen FLOPs'ta PPO λ=0.4 +2.1 [+0.3,+4.0], λ=0.8 +2.6 [+0.6,+4.7] puan, ancak basit sezgisel (τ=0.5) de +3.3 [+0.2,+6.2]: kazanç uyarlanır K'dan geliyor, PPO'nun sezgisele üstünlüğü kanıtlanmadı; 3 seed'de λ=0.4 işareti pozitif (+1.3…+2.1), λ=0.8 seed'e bağlı. ScienceQA-IMG'de AI2D politikaları taşınmıyor (−2.5…−3.2); ScienceQA'da 5 seed yeniden eğitilince +1.0…+8.7 (karışım zayıf referans; Top-3'ü eşliyor ya da %14–22 daha az FLOPs ile 1–3.6 puan geride), sezgisel tutarsız. Kazanç küçük, göreve özgü yeniden eğitim gerektiriyor. TOPSIS bileşeni DeepSeek'te ek fayda sağlamadı.

**Sabit asimetrik K baseline'ı (`scripts/run_static.sh`, 21 Ekim):** AI2D kazancı sabit (K_prefill=2, K_decode=4) kuralıyla açıklanıyor (PPO − Top-K+sabit zarf: +0.05…+0.75, GA'lar 0'ı içeriyor). ScienceQA'da öğrenilmiş politikalar zarfın +1…+8 puan üstünde kalıyor (5 seed'den 4'ünde GA 0'ı dışlıyor): soru başına K seçimi yalnızca burada kanıtlı.

**A-OKVQA (3. veri seti, `scripts/run_aok.sh`, 22 Ekim):** PPO üç seed'de de mevcut Top-K noktalarını yeniden buluyor (Top-3 ×2, Top-2 ×1); Top-K+sabit zarfa göre −0.9…−0.1 puan, GA'lar 0'ı içeriyor. Soru başına öğrenilmiş K yalnızca ScienceQA-IMG'de (3 veri setinden 1'inde) kazanç sağlıyor.
