# DecisionMoE — Claude Code Implementation Briefing

> **Bu doküman ne içindir:** Claude Code agent'ının projeyi eksiksiz anlayıp implementation'a başlayabilmesi için hazırlanmış tam referans belgesidir.
> **Mevcut durum:** Tüm teorik, mimari ve organizasyonel kararlar tamamlanmıştır. Yapılacak tek şey kodu yazmaktır.
> **Kesinlikle değiştirilmeyecekler:** Dosya yapısı, JSON formatları, matematiksel formülasyonlar, kriter tanımları, reward katsayıları, split oranları.

---

## 1. Proje Özeti

**Tam Ad:** DecisionMoE: Multi-Criteria Reinforcement Learning for Context-Aware Expert Routing in Multimodal Large Language Models

**Hedef:** Springer Nature IJCV Special Issue (Multimodal LLMs for Unified Comprehension and Generation) — **Deadline: 1 Kasım 2026**

**Çekirdek Fikir:**
MoE modellerindeki expert routing'i iki aşamalı hiyerarşik karar sistemi olarak formüle etmek:
1. **PPO** (Proximal Policy Optimization) — her token için 7 kriterin ağırlıklarını ($w_t$) ve kaç expert seçileceğini ($K_t$) öğrenir
2. **TOPSIS** (Multi-Criteria Decision Making) — $w_t$ ağırlıklarını kullanarak 4 expert arasından en iyileri ranklar

**En yakın rakipler ve ayrışma:**
- **MoE-GRPO** (arXiv 2603.24984, Mart 2026): GRPO ile multimodal MoE routing optimize ediyor — ama tek kriter, MCDM yok, K sabit
- **Ada-K Routing** (ICLR 2025): PPO ile Dynamic-K — ama tek kriter, MCDM yok, text-only

DecisionMoE'nin ayrışması: Bu kombinasyon (RL criteria weighting + TOPSIS ranking + multimodal + Dynamic-K) literatürde **yoktur**.

---

## 2. Backbone Model

| Özellik | Değer |
|---|---|
| Model | MoE-LLaVA-Phi2-2.7B×4-Top2 |
| HuggingFace ID | `LanguageBind/MoE-LLaVA-Phi2-2.7B-4e` |
| Expert sayısı | 4 |
| Başlangıç Top-K | 2 (Dynamic-K ile öğrenilecek) |
| Toplam parametre | 5.3B |
| VRAM ihtiyacı | ~11GB |
| Backbone durumu | **Frozen** — sadece routing katmanı train edilir |
| Training makinesi | RTX 5060 (12GB VRAM) |
| Inference/analiz | RTX 4070 laptop (8GB VRAM) |

**Backbone yükleme kuralı:** Tüm backbone ağırlıkları dondurulur. Yalnızca şunlar train edilir:
- PPO policy network (routing controller)
- Linear projection katmanı (h_t: 2560 → 64)
- Expert prototype vektörleri (bir kere hesaplanır, frozen kalır)

---

## 3. Matematiksel Formülasyon

### 3.1 TOPSIS Katmanı

Her routing adımında $N=4$ expert, $M=7$ kriter için:

**Karar matrisi:**
$$D \in \mathbb{R}^{4 \times 7}$$

**Kriter tanımları:**

| j | Kriter | Tür | Ölçüm Yöntemi |
|---|---|---|---|
| 1 | Task Utility | Benefit | Cosine similarity: input hidden state ↔ expert prototype vektörü |
| 2 | Modality Match | Benefit | Expert'in o modality için training-time aktivasyon oranı |
| 3 | Specialization | Benefit | Expert aktivasyon entropisinin tersi: $1 - H(\text{activation history})$ |
| 4 | Reliability | Benefit | ECE (Expected Calibration Error) tersi: $1 - \text{ECE}_i$ |
| 5 | Latency | Cost | Expert forward pass süresi (ms, runtime profiling) |
| 6 | FLOPs | Cost | Expert parametre sayısı × aktivasyon yoğunluğu (normalize) |
| 7 | Load Balance | Cost | Son T=100 token'daki expert utilization oranı |

**Benefit kriterleri:** $\mathcal{B} = \{1, 2, 3, 4\}$
**Cost kriterleri:** $\mathcal{C} = \{5, 6, 7\}$

**Adım 1 — L2 Normalizasyon:**
$$r_{ij} = \frac{d_{ij}}{\sqrt{\sum_{i=1}^{4} d_{ij}^2}}$$

**Adım 2 — Ağırlıklı Normalizasyon ($w_t$ PPO'dan gelir):**
$$v_{ij} = w_j \cdot r_{ij}, \quad \sum_{j=1}^{7} w_j = 1$$

**Adım 3 — İdeal Çözümler:**
$$A^+_j = \begin{cases} \max_i v_{ij} & j \in \mathcal{B} \\ \min_i v_{ij} & j \in \mathcal{C} \end{cases}, \quad A^-_j = \begin{cases} \min_i v_{ij} & j \in \mathcal{B} \\ \max_i v_{ij} & j \in \mathcal{C} \end{cases}$$

**Adım 4 — Öklid Mesafeleri:**
$$d_i^+ = \sqrt{\sum_{j=1}^{7}(v_{ij} - A^+_j)^2}, \quad d_i^- = \sqrt{\sum_{j=1}^{7}(v_{ij} - A^-_j)^2}$$

**Adım 5 — TOPSIS Skoru:**
$$S_i = \frac{d_i^-}{d_i^+ + d_i^-}, \quad S_i \in [0,1]$$

**Adım 6 — Expert Seçimi:**
$$\text{Selected} = \text{Top-}K_t\bigl(\{S_i\}_{i=1}^{4}\bigr)$$

---

### 3.2 PPO State Vektörü

$$s_t = [h_t,\ m_t,\ d_t,\ u_t,\ p_t,\ l_t,\ c_t]$$

**Gerçek PPO girişi** ($h_t$ projection sonrası):
$$\tilde{s}_t \in \mathbb{R}^{74} = [\underbrace{\text{Linear}_{2560\to64}(h_t)}_{64},\ \underbrace{m_t}_{3},\ \underbrace{d_t}_{1},\ \underbrace{u_t}_{1},\ \underbrace{p_t}_{4},\ \underbrace{l_t}_{4},\ \underbrace{c_t}_{1}]$$

| Bileşen | Boyut | Hesaplama |
|---|---|---|
| $h_t$ | 2560 → 64 | Backbone hidden state, Linear projection ile indirgenir |
| $m_t$ | 3 | Modality one-hot: `[1,0,0]`=text, `[0,1,0]`=image, `[0,0,1]`=mixed |
| $d_t$ | 1 | Gating entropy: $-\sum_i p_i \log p_i$, $p_i=\text{softmax}(W_g h_t)$ |
| $u_t$ | 1 | Uncertainty: $1 - \max_i p_i$ |
| $p_t$ | 4 | Reasoning phase one-hot: perception / question / reasoning / generation |
| $l_t$ | 4 | Expert load: son T=100 token'da her expert'in aktivasyon oranı |
| $c_t$ | 1 | Kalan FLOPs budget: $B_\text{remaining}/B_\text{total}$ |

**Reasoning phase tespiti (heuristic):**
- `perception`: Vision encoder token bölgesi
- `question`: Soru token'ları (soru işareti içeren segment)
- `reasoning`: Orta segment (varsayılan)
- `generation`: `<answer>` token'ından sonrası

---

### 3.3 PPO Action Space

**Alternatif B (birincil):** $a_t = [w_t, K_t]$
- $w_t \in \Delta^6$ (7-boyutlu simplex, $\sum w_j = 1$, $w_j \geq 0$)
- $K_t \in \{1, 2, 3, 4\}$ — **hard constraint: $K_t \geq 1$** (reward hacking koruması)

**Alternatif A (fallback):** Sadece $w_t$, $K_t = 2$ sabit

---

### 3.4 Reward Fonksiyonu

$$R_t = \lambda_1 A_t - \lambda_2 F_t - \lambda_3 L_t - \lambda_4 I_t - \lambda_5 H_t$$

| Terim | Sembol | Formül | Başlangıç $\lambda$ |
|---|---|---|---|
| Task accuracy | $A_t$ | $\mathbb{1}[\hat{y}=y^*]$ veya normalize CIDEr | $\lambda_1 = 1.00$ |
| FLOPs maliyeti | $F_t$ | $\frac{\sum_{i\in\text{selected}}\text{FLOPs}(E_i)}{\text{FLOPs}_\text{baseline}}$ | $\lambda_2 = 0.10$ |
| Latency | $L_t$ | $\tau_t / \tau_\text{baseline}$ | $\lambda_3 = 0.05$ |
| Load imbalance | $I_t$ | $\text{Std}(l_t)$ | $\lambda_4 = 0.10$ |
| Hallucination | $H_t$ | $\mathbb{1}[\text{hallucination detected}]$ | $\lambda_5 = 0.20$ |

**Kritik ayrım:**
- $\lambda$ katsayıları → araştırmacı tarafından sabit belirlenir (global training objective)
- $w_t$ ağırlıkları → PPO tarafından her token için dinamik üretilir (context-dependent)

---

## 4. Ablation Tablosu — 7 Variant

| Variant | RL | MCDM | Dyn-$w$ | Dyn-$K$ | Task Utility |
|---|:---:|:---:|:---:|:---:|:---:|
| Standard Top-K | ✗ | ✗ | ✗ | ✗ | — |
| Fixed MCDM (no utility) | ✗ | ✓ | ✗ | ✗ | ✗ |
| Fixed MCDM (cosine proxy) | ✗ | ✓ | ✗ | ✗ | ✓ |
| PPO + Fixed-K, no utility | ✓ | ✓ | ✓ | ✗ | ✗ |
| PPO + Fixed-K, cosine proxy | ✓ | ✓ | ✓ | ✗ | ✓ |
| **DecisionMoE-A** (Full, no utility) | ✓ | ✓ | ✓ | ✓ | ✗ |
| **DecisionMoE-B** (Full, cosine proxy) | ✓ | ✓ | ✓ | ✓ | ✓ |

Her variant `experiments/ablation/variant_*.yaml` config dosyasıyla çalıştırılır.

---

## 5. Dataset

Dataset ekip arkadaşı tarafından üretilmektedir. Implementation başladığında `data/decisionmoe_dataset/` altında hazır olacak.

**Kaynak görüntüler:** COCO 2017, AI2D, GQA, TextVQA (~2000 görüntü)

**Soru tipleri:**

| Tip | Açıklama | Routing Etkisi |
|---|---|---|
| T1 | Visual Perception — doğrudan görsel tanıma | Perception-specialist expert |
| T2 | Spatial Reasoning — nesneler arası konum | Reasoning-specialist expert |
| T3 | Multi-hop QA — iki adımlı çıkarım | Hem perception hem reasoning |
| T4 | Hallucination Trap — görselde olmayan şeyi sorar | Reliability-sensitive routing |
| T5 | OCR + Reasoning — görüntüdeki metin + mantık | Specialization-sensitive routing |

**Sabit JSON formatı** (`questions.json`):
```json
{
  "id": "coco_000001_T1",
  "image_id": "000001",
  "image_path": "images/coco/000001.jpg",
  "source": "coco",
  "question_type": "T1",
  "question": "How many people are in this image?",
  "answer": "3",
  "language": "en"
}
```

**Split:** %70 train / %15 val / %15 test (stratified, soru tipi ve kaynak bazlı)

---

## 6. Repo Yapısı — Sabit ve Final

```
decisionmoe/
│
├── configs/
│   ├── model.yaml          # Backbone, expert sayısı, VRAM ayarları
│   ├── ppo.yaml            # PPO hyperparameter'ları
│   ├── topsis.yaml         # Kriter tanımları, benefit/cost mask
│   └── training.yaml       # Batch size, lr, epoch, wandb proje adı
│
├── data/
│   └── decisionmoe_dataset/
│       ├── images/
│       │   ├── coco/
│       │   ├── ai2d/
│       │   ├── gqa/
│       │   └── textvqa/
│       ├── questions.json
│       └── splits/
│           ├── train.json
│           ├── val.json
│           └── test.json
│
├── src/
│   ├── backbone/
│   │   ├── __init__.py
│   │   ├── loader.py        # MoE-LLaVA yükleme, backbone freeze
│   │   └── prototype.py     # Expert prototype vektör hesabı (cosine proxy)
│   │
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── topsis.py        # TOPSIS katmanı (saf PyTorch)
│   │   ├── criteria.py      # 7 kriterin online hesabı
│   │   └── phase.py         # Reasoning phase tespiti (heuristic)
│   │
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── ppo.py           # PPO training loop
│   │   ├── policy.py        # Policy network: R^74 → (w_t ∈ Δ^6, K_t ∈ {1,2,3,4})
│   │   └── reward.py        # Reward fonksiyonu, λ katsayıları
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # questions.json okuma, PyTorch Dataset
│   │   └── collator.py      # Batch hazırlama, padding
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py       # Accuracy, hallucination, FLOPs sayacı
│       └── logger.py        # wandb wrapper
│
├── experiments/
│   ├── ablation/
│   │   ├── variant_standard_topk.yaml
│   │   ├── variant_fixed_mcdm_no_utility.yaml
│   │   ├── variant_fixed_mcdm_cosine.yaml
│   │   ├── variant_ppo_fixedk_no_utility.yaml
│   │   ├── variant_ppo_fixedk_cosine.yaml
│   │   ├── variant_decisionmoe_a.yaml
│   │   └── variant_decisionmoe_b.yaml
│   │
│   ├── baseline/
│   │   └── run_topk_baseline.py
│   │
│   └── results/
│       ├── ablation_scores.json      # T1 tablosu ham verisi
│       ├── efficiency_metrics.json   # T2 tablosu ham verisi
│       ├── expert_utilization.json   # T3 tablosu ham verisi
│       ├── criteria_weights.json     # T4 tablosu ham verisi
│       └── routing_logs/
│
├── paper/
│   ├── style.py                      # Sabit Matplotlib/Seaborn stil ayarları
│   ├── figures/
│   │   ├── f1_architecture/
│   │   │   └── architecture.png      # Manuel çizim (draw.io) — kod üretmez
│   │   ├── f2_reward_curve.png
│   │   ├── f3_dynamic_k_distribution.png
│   │   ├── f4_expert_activation_heatmap.png
│   │   ├── f5_criteria_weight_radar.png
│   │   ├── f6_ablation_bar_chart.png
│   │   └── f7_hallucination_load_scatter.png
│   │
│   └── tables/
│       ├── t1_main_results.tex
│       ├── t2_efficiency.tex
│       ├── t3_expert_utilization.tex
│       ├── t4_criteria_weights.tex
│       └── t5_dataset_statistics.tex
│
├── scripts/
│   ├── train.py                      # Ana training scripti
│   ├── evaluate.py                   # Benchmark evaluation
│   ├── compute_prototypes.py         # Expert prototype hesabı (bir kere çalışır)
│   └── generate_paper_materials.py   # Tüm figür ve tabloları üretir
│
├── tests/
│   ├── test_topsis.py
│   ├── test_criteria.py
│   ├── test_reward.py
│   ├── test_ppo.py
│   ├── test_dataset.py
│   └── test_metrics.py
│
├── requirements.txt
├── README.md
└── .env.example
```

---

## 7. Kod Standartları

### Framework ve Bağımlılıklar
```
Python          3.12
PyTorch         2.3+
Transformers    4.40+   (HuggingFace — backbone)
wandb           0.17+   (experiment tracking)
pytest          8.0+    (test)
numpy           1.26+
scipy           1.13+   (TOPSIS mesafe hesabı)
matplotlib      3.8+    (figür üretimi)
seaborn         0.13+   (heatmap, advanced plots)
pyyaml          6.0+    (config okuma)
python-dotenv   1.0+    (API key yönetimi)
```

### Genel Kurallar
- Dosya başında modül amacını açıklayan docstring zorunlu
- Fonksiyon isimleri `snake_case`, sınıf isimleri `PascalCase`
- Sabitler `UPPER_SNAKE_CASE` ve her zaman `configs/*.yaml`'dan okunur — **kod içine gömülmez**
- Magic number yok — örn. `K_MIN = 1`, `K_MAX = 4`
- Type hint: public API'larda zorunlu, private method'larda (`_` prefix) opsiyonel
- Tensor shape'leri docstring'de belirtilir

```python
def compute_topsis_scores(
    decision_matrix: torch.Tensor,   # (N_experts=4, N_criteria=7)
    weights: torch.Tensor,            # (N_criteria=7,) — PPO'dan gelir, sum=1
    benefit_mask: torch.Tensor,       # (N_criteria=7,) bool — True=benefit
) -> torch.Tensor:                    # (N_experts=4,) TOPSIS skoru ∈ [0,1]
    """TOPSIS skorlarını hesaplar. Yüksek skor = ideal çözüme yakın."""
```

### Cihaz Yönetimi
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Kodda hardcode "cuda" veya "cpu" kullanılmaz
```

### Config Yönetimi
```python
import yaml
with open("configs/topsis.yaml") as f:
    config = yaml.safe_load(f)
# Training başında tüm config wandb'a loglanır
wandb.config.update(config)
```

### wandb Loglama Standartları
```python
wandb.init(
    project="decisionmoe",
    name=f"{variant_name}_{timestamp}",
    config=full_config,
    tags=[variant_name, "ablation"]
)

# Her step'te loglanacaklar (sabit key isimleri):
wandb.log({
    "train/reward": reward,
    "train/accuracy": accuracy,
    "train/flops_ratio": flops_ratio,
    "train/load_imbalance": imbalance,
    "train/hallucination_rate": hallucination,
    "routing/mean_K": mean_k,
    "routing/w_task_utility": w_t[0],
    "routing/w_modality_match": w_t[1],
    "routing/w_specialization": w_t[2],
    "routing/w_reliability": w_t[3],
    "routing/w_latency": w_t[4],
    "routing/w_flops": w_t[5],
    "routing/w_load_balance": w_t[6],
    "routing/expert_0_load": expert_load[0],
    "routing/expert_1_load": expert_load[1],
    "routing/expert_2_load": expert_load[2],
    "routing/expert_3_load": expert_load[3],
})
```

---

## 8. Test Standartları

Her `src/` modülü için karşılık gelen `tests/test_*.py` dosyası yazılır.

### Zorunlu Test Kapsamı

| Test Dosyası | Zorunlu Test Senaryoları |
|---|---|
| `test_topsis.py` | L2 norm sonrası $\|r_{\cdot j}\|=1$; $S_i \in [0,1]$; en iyi expert doğru seçiliyor; benefit/cost yönleri doğru |
| `test_criteria.py` | Her kriterin output aralığı ve torch.Tensor tipi; online güncelleme tutarlılığı |
| `test_reward.py` | $K_t=0$ imkansız; $\lambda$ katsayıları toplamı ve etkileri; bilinen input → bilinen reward |
| `test_ppo.py` | $w_t$ toplamı $= 1.0$ (±1e-6); $K_t \in \{1,2,3,4\}$; policy output shape doğru |
| `test_dataset.py` | JSON format doğrulaması; split sızıntısı yok (train/val/test ID'leri kesişmiyor) |
| `test_metrics.py` | Bilinen cevaplar için bilinen metrik değerleri |

```bash
pytest tests/ -v                  # tümü
pytest tests/test_topsis.py -v    # tek modül
pytest tests/ --cov=src           # coverage raporu
```

---

## 9. Paper Materyalleri

### `paper/style.py` — Sabit, Değiştirilmez
```python
PAPER_RC = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6.5, 4.0),  # IJCV single-column
}

COLORS = {
    "decisionmoe_b": "#2E86AB",  # Ana model
    "decisionmoe_a": "#A23B72",  # Utility-free
    "baseline":      "#F18F01",  # Standard Top-K
    "ablation":      "#C73E1D",  # Diğer variant'lar
    "neutral":       "#6B6B6B",
}
```

### Figür Listesi

| Dosya | İçerik | Kaynak Veri |
|---|---|---|
| `f1_architecture.png` | Genel pipeline diyagramı | Manuel (draw.io) — **kod üretmez** |
| `f2_reward_curve.png` | Training reward, accuracy, FLOPs tradeoff | wandb / `results/` JSON |
| `f3_dynamic_k_distribution.png` | Soru tipine göre K dağılımı (histogram) | `routing_logs/` |
| `f4_expert_activation_heatmap.png` | Soru tipi × Expert aktivasyon (4×5 heatmap) | `expert_utilization.json` |
| `f5_criteria_weight_radar.png` | T1-T5 için $w_t$ profili (radar chart) | `criteria_weights.json` |
| `f6_ablation_bar_chart.png` | 7 variant × benchmark (grouped bar) | `ablation_scores.json` |
| `f7_hallucination_load_scatter.png` | POPE skoru vs load imbalance (scatter) | `ablation_scores.json` + `efficiency_metrics.json` |

### Tablo Listesi

| Dosya | İçerik |
|---|---|
| `t1_main_results.tex` | 7 variant × 5 benchmark — ana sonuç tablosu |
| `t2_efficiency.tex` | FLOPs, latency, VRAM karşılaştırması |
| `t3_expert_utilization.tex` | Expert aktivasyon oranları (4×5 matris) |
| `t4_criteria_weights.tex` | Soru tipine göre ortalama $w_t$ değerleri |
| `t5_dataset_statistics.tex` | Soru tipi / kaynak dağılımı, split boyutları |

### `generate_paper_materials.py` Çalıştırma
```bash
python scripts/generate_paper_materials.py --results_dir experiments/results/
# Çıktı: paper/figures/*.png + paper/tables/*.tex
```

---

## 10. Git Kuralları

### Branch Stratejisi
```
main              → çalışan, test edilmiş kod
dev               → aktif geliştirme
feature/backbone
feature/topsis
feature/ppo
feature/reward
experiments/ablation-a
experiments/ablation-b
```

### Commit Formatı
```
[modül] kısa açıklama

Örnekler:
feat/routing: TOPSIS L2 normalizasyon implementasyonu
feat/rl: PPO policy network, w_t softmax + K_t sampling
fix/reward: K_t=0 hard constraint eklendi
test/topsis: benefit/cost yönlendirme edge case
exp/baseline: MMBench Standard Top-K baseline çalıştırıldı
```

### `.gitignore` Zorunluları
```
.env
data/decisionmoe_dataset/images/
experiments/results/routing_logs/
wandb/
__pycache__/
*.pyc
*.pth          # Model checkpoint'leri ayrı paylaşılır
```

---

## 11. Modül Yazım Sırası

Bağımlılık sırasına göre — **bu sıra değiştirilmez:**

```
1. src/backbone/loader.py          → MoE-LLaVA yükle, freeze, smoke test
2. scripts/compute_prototypes.py   → Expert prototype vektörlerini hesapla, kaydet
3. src/routing/criteria.py         → 7 kriterin online hesabı
4. src/routing/topsis.py           → TOPSIS katmanı (pytest ile doğrula)
5. src/routing/phase.py            → Reasoning phase heuristic
6. src/rl/reward.py                → Reward fonksiyonu (pytest ile doğrula)
7. src/rl/policy.py                → Policy network: R^74 → (w_t, K_t)
8. src/rl/ppo.py                   → PPO training loop
9. src/data/dataset.py             → Dataset + DataLoader
10. src/data/collator.py           → Batch collation
11. src/utils/metrics.py           → Evaluation metrikleri
12. src/utils/logger.py            → wandb wrapper
13. scripts/train.py               → Hepsini birleştir, ilk training run
14. experiments/baseline/...       → Standard Top-K baseline
15. scripts/evaluate.py            → Benchmark evaluation
16. paper/style.py                 → Figür stil ayarları
17. scripts/generate_paper_materials.py → Figür ve tablo üretimi
```

---

## 12. Haftalık Zaman Planı

| Hafta | Görev | Hedef Çıktı |
|---|---|---|
| **1–2** | Backbone kurulum, prototype hesabı, baseline | `loader.py`, `prototype.py`, Top-K baseline skorları |
| **3–4** | TOPSIS + criteria implementasyonu | `topsis.py`, `criteria.py`, pytest geçiyor |
| **5–6** | PPO controller | `policy.py`, `ppo.py`, `reward.py`, ilk training run |
| **7–8** | Full model + tüm benchmark evaluation | DecisionMoE-A ve B sonuçları, tüm ablation |
| **9** | Analiz, figür ve tablo üretimi | `paper/figures/`, `paper/tables/` hazır |
| **10** | Paper yazımı, IJCV template, submission | Submission |

---

## 13. Başlangıç Kontrol Listesi

Claude Code'un ilk yapacakları:

- [ ] Repo klasör yapısını oluştur (Section 6'daki tam yapı)
- [ ] `requirements.txt` oluştur (Section 7'deki bağımlılıklar)
- [ ] `.env.example` oluştur (`WANDB_API_KEY=`, `HF_TOKEN=`)
- [ ] `configs/*.yaml` dosyalarını oluştur (boş şablonlar, doldurmak için hazır)
- [ ] `paper/style.py` oluştur (Section 9'daki sabit değerlerle)
- [ ] `src/backbone/loader.py`'dan başla — Section 11'deki sırayı takip et

---

*DecisionMoE Araştırma Ekibi — Ağustos 2026*
*Tüm teorik ve mimari kararlar tamamlanmıştır. Bu doküman implementation için tek referanstır.*
