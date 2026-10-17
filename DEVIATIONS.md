# Briefing'den sapmalar ve belirsiz bırakılan kararlar

Makalenin Yöntem ve Sınırlılıklar bölümleri için kayıt. Her madde kodda nerede olduğunu gösterir.

## A. Briefing'deki tutarsızlıklar
| # | Konu | Briefing | Uygulama | Dosya |
|---|---|---|---|---|
| A1 | PPO durum boyutu | ℝ⁷⁴ | **78** (64+3+1+1+4+4+1; bileşenler tabloda tek tek verilmiş, toplam yanlış) | `configs/ppo.yaml`, `src/rl/policy.py` |

## B. Briefing'in belirsiz bıraktığı tanımlar (bizim seçimimiz)
| # | Konu | Seçim | Dosya |
|---|---|---|---|
| B1 | Expert prototype | MoE katmanı girişinin, o experti top-1 seçen token'lar üzerindeki ortalaması (katman başına, 400 train örneği) | `src/backbone/prototype.py` |
| B2 | Specialization | 1 − H(P(phase\|expert))/log 4 (Laplace yumuşatmalı, online) | `src/routing/criteria.py` |
| B3 | FLOPs kriteri | expert param payı × son 100 token'daki aktivasyon yoğunluğu | `src/routing/criteria.py` |
| B4 | Task Utility | (cos+1)/2, TOPSIS L2 normu için negatif değer olmasın | `src/routing/criteria.py` |
| B5 | Tasarım A | Tek policy; adım başına bir (w_t,K_t) tüm MoE katmanlarına uygulanır. Karar matrisi D katman başına hesaplanır | `src/routing/router.py` |
| B6 | Routing adımı | Prefill chunk'ı = 1 adım, her üretilen token = 1 adım; chunk içi load/specialization geçmişi chunk başındaki değerden okunur (yaklaşıklık) | `src/routing/router.py` |
| B7 | Karışım ağırlığı | Seçilen expertlerin backbone gate olasılıkları yeniden normalize edilir | `src/routing/router.py` |
| B8 | Reward | Adım-yerel maliyetler her adıma; doğruluk ve hallucination yalnızca bölüm sonuna; GAE(γ=1, λ=0.95). **FLOPs terimi token ağırlıklıdır** (adım payı = n_tokens/Σn_tokens), bu yüzden λ₂ = 1,0 (adım bazlı tanımda 0,1'di). Bkz. F1 | `src/rl/ppo.py` |
| B9 | Latency referansı | prefill ve decode için ayrı τ_baseline | `scripts/train.py` |
| B10 | Hallucination H_t | Yalnızca T4 sorularında, olumsuz yanıt vermemek | `src/utils/metrics.py` |
| B11 | "Sadece PPO" kolu (DIRECT) | p'_e ∝ p_gate,e · u_e; u_t ∈ Δ³ PPO'dan, TOPSIS yok | `src/routing/router.py` |

## C. Ölçülmemiş / etkisiz kriterler (makalede açıkça belirtilmeli)
| # | Konu | Durum |
|---|---|---|
| C1 | Reliability (1−ECE) | **Ölçülmedi**, tüm expertler için sabit 1.0 → TOPSIS'te ayırt edici değil |
| C2 | Latency (ms) | **Ölçülmedi**, sabit tablo. Expertler aynı mimaride olduğundan zaten expertler arası fark beklenmez |
| C3 | FLOPs | Expert parametreleri eşit; kriter yalnızca yük üzerinden değişir |
| C4 | Task Utility ayırt ediciliği | Prototype'lar birbirine çok benzer (ortalama cosine 0,6–0,9; katman 15 hariç) |

## D. Donanım ve deney koşulları
| # | Konu | Durum |
|---|---|---|
| D1 | GPU | RTX 5060 **8 GB** (briefing 12 GB varsayıyor) |
| D2 | Kuantizasyon | **nf4**. int8 bitsandbytes 0.50 + torch 2.11 ile üretimde CUDA hatası verdi; fp16 8 GB'a sığmıyor |
| D3 | Baseline "Top-K" | Deterministik top-2 (kapasite sınırı yok). DeepSpeed'in orijinal gate'i 2. experti Gumbel gürültüsüyle örnekler; birebir eşleşme testi bu gürültü kapatılarak yapıldı |
| D4 | Ortam yamaları | `venv_moe`: transformers 4.37.0, accelerate 0.26.1, bitsandbytes `ops.py:233` `.view`→`.reshape` yaması (`scripts/setup_moe_env.sh`) |
| D5 | Seed | DecisionMoE-B için **3 seed** (42, 1, 2); diğer PPO variant'ları tek seed (42). Baseline ve sabit-MCDM variant'ları deterministik |
| D6 | Veri | Sorular ve cevaplar Qwen2.5-VL ile üretildi, insan doğrulaması yok. 931/2000 görüntüde bir soru tipi eksik (T2 zayıf: 1072) |
| D7 | Benchmark | Standart benchmark olarak yalnızca **POPE** (3×1000 soru); GQA/TextVQA test koşulmadı |
| D8 | Latency ölçümü | Ayrı kontrollü benchmark (`benchmark_latency.py`, ısınma + 3 tekrar, n=100); yalnızca 5 variant ölçüldü, ms/soru GPU-senkronlu. Router profili `profile_router.py` |

## E. Temel bulgular (sapma değil, sonuç)
- Expert seçimi bu backbone'da doğruluğu değiştirmiyor (`experiments/results/sensitivity_val.json`): 11 konfigürasyon 0,380–0,405.
- Aynı katmandaki 4 expert ağırlığı yalnızca ~%3–4 farklı.
- Görüntü kullanılıyor: gerçek 0,405 vs boş görüntü 0,273.

## F. Düzeltilen tasarım hatası (run2 → run3)
| # | Konu | Ayrıntı |
|---|---|---|
| F1 | **Adım bazlı FLOPs ödülü sömürüldü (reward hacking)** | İlk Dynamic-K koşularında (run2) FLOPs terimi adım başına eşit sayılıyordu; ~600 token işleyen prefill adımı da "tek adım"dı. Politikalar prefill'de **K=4**, decode'da K≈1 seçti. Rapor edilen "FLOPs 0,33" adım ortalamasıydı; **token ağırlıklı gerçek maliyet ≈ 0,99** (Top-2: 0,50) ve gecikme 1,21× idi. Run2 sonuçları `experiments/results/archive_run2_stepflops/` altında kanıt olarak saklanıyor. Düzeltme: ödül ve değerlendirmede FLOPs token ağırlıklı, `flops_fraction_step` yalnızca karşılaştırma için |
| F2 | Sabit-K variant'lar | Etkilenmedi (K sabit, token ağırlıklı oran = K/E); yeniden koşulmadı |

## G. Veri kalitesi denetimi (model destekli; insan değil)
| # | Bulgu | Kanıt |
|---|---|---|
| G1 | 200 test sorusunun denetimi: kesin yanlış referans **%7,5** [4,6–12,0], doğrulanamayan **%40**, sorunlu soru **%26**. T2 (konum) en zayıf: %63 sorunlu | `experiments/results/audit_sheet.csv` (etiketleyen: dil modeli, görüntüler görülerek; sayma ve küçük yazıda hata olabilir) |
| G2 | Modelin kesin değerlendirilebilen cevaplarında anlamca doğruluk **%66**; katı eşleşme doğruluğu 0,39 | aynı tablo |
| G3 | **T4 hallucination oranı şişmiş:** hallucination sayılan 39 sorunun **17'sinde (%44) aranan nesne/metin görüntüde gerçekten var** (bozuk tuzak), 11'i belirsiz, yalnızca 11'i gerçek hallucination. Düzeltilmiş oran: ham 0,136 → bozuklar çıkarıldı 0,082 → bozuk+belirsiz çıkarıldı 0,041 (`standard_topk`) | `experiments/results/t4_audit.json`, `scripts/analyze_results.py` |
| G4 | Sınırlama | Yalnızca bir variant'ın hallucination saydığı sorular denetlendi; tüm 273 T4 soru ve tüm tipler tek tek doğrulanmadı. Etiketleyici tek model, hakemler arası uyum ölçülmedi |

## D9 — Uzmanlaşma deneyi (DM_TAG=spec, 17 Ekim)
Backbone'a expert-LoRA eklendi (`src/backbone/expert_lora.py`, `DM_EXPERT_LORA` ortam değişkeniyle yüklenir); orijinal sonuçlar etkilenmez.
Alanlar `src/data/domains.py`; eğitim `scripts/train_experts.py` (yalnızca train split); alan prototipleri `scripts/compute_domain_prototypes.py`
(`artifacts/prototypes_spec.pt`); `DM_TAG` dosya adlarına `_spec` ekler. Sınırlama: LoRA ve test aynı dağılımdan; PPO yalnızca 80 güncelleme ve tek seed; decisionmoe_b bu profilde koşulmadı.
