# DecisionMoE

**MoE tabanlı görsel-dil modellerinde çok kriterli karar verme (TOPSIS) ve pekiştirmeli öğrenme (PPO) ile expert yönlendirme**

`Python 3.12` · `PyTorch` · `MoE-LLaVA-Phi2-2.7B×4` · `TOPSIS + PPO` · `Dynamic-K`

**İçindekiler:** [Bulgular](#1-ne-yapıyor-ne-buldu) · [Depo yapısı](#2-depo-yapısı) · [Kurulum](#3-kurulum) · [Süreç](#4-baştan-sona-süreç) · [Testler](#5-testler) · [Tekrarlanabilirlik](#6-tekrarlanabilirlik-notları)

**Hızlı bakış**

| | |
|---|---|
| Soru | Expert seçimi doğruluğu gerçekten etkiliyor mu? |
| Yöntem | 5 kriter → TOPSIS skoru; ağırlık ve K (1–4) PPO politikasından |
| Veri | 8803 soru, 5 soru tipi, COCO/AI2D/GQA/TextVQA + POPE doğrulaması |
| Sonuç | Backbone expert seçimine duyarsız; K=1, K=2 ile eşit doğrulukta yarı FLOPs |
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

Makalenin konumu: **negatif/analiz sonucu** — routing'in doğruluğa etkisi yok, ödül tasarımı kritik, RL K=1'e yakınsıyor.

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
`scripts/analyze_specialization.py` (özet + `paper/tables/t6_specialization.tex`). Sonuç: bkz. `paper/draft.md` §5.7.
