# DecisionMoE — Sentetik Dataset Üretim Kılavuzu

> **Bu doküman kime yazıldı:** Dataset üretiminden sorumlu ekip üyesine.
> **Amaç:** Claude Code agent'ına vererek süreci planlayıp uygulayabilmesi için tam bir referans.
> **Kritik not:** Dosya yapısı, split oranları ve JSON formatları **kesinlikle değiştirilmemelidir.** Ana geliştirici bu yapıya göre kod yazacaktır.

---

## 1. Projenin Bağlamı

Bu dataset, **DecisionMoE** adlı bir araştırma projesinin eğitim ve test verisidir.

DecisionMoE, MoE-LLaVA-Phi2-2.7B modelinin expert routing mekanizmasını iyileştiren bir sistemdir. Model 4 expert içerir ve her token işlenirken bu 4 expert arasından 2'si seçilir. Sistemimiz bu seçimi:

- **TOPSIS** (çok kriterli karar yöntemi) ile ranklar
- **PPO** (pekiştirmeli öğrenme) ile kriter ağırlıklarını dinamik öğrenir

Dataset'in görevi: Farklı soru tipleri farklı expert kombinasyonlarını optimal kılmalı. Bunu sağlamadan PPO routing'i öğrenemez — hep aynı 2 expert'e gider.

---

## 2. Değiştirilmemesi Gereken Kurallar

> ⚠️ Aşağıdaki kararlar ana geliştirici tarafından verilmiştir. Claude Code agent'ın bunları **kesinlikle değiştirmemesi** gerekir.

### 2.1 Dosya Yapısı — Sabit

```
decisionmoe_dataset/
├── images/
│   ├── coco/              # COCO 2017 görüntüleri
│   ├── ai2d/              # AI2D diyagram görüntüleri
│   ├── gqa/               # GQA görüntüleri
│   └── textvqa/           # TextVQA görüntüleri
├── raw/
│   └── generated_raw.json # LLM'den ham çıktı (filtrelenmemiş)
├── questions.json          # Ana dataset dosyası (filtreli, final)
├── routing_oracle.json     # Expert aktivasyon logları (opsiyonel)
└── splits/
    ├── train.json          # %70
    ├── val.json            # %15
    └── test.json           # %15
```

### 2.2 JSON Formatı — Sabit

`questions.json` içindeki her kayıt tam olarak şu formatta olmalıdır:

```json
{
  "id": "coco_000001_T1",
  "image_id": "000001",
  "image_path": "images/coco/000001.jpg",
  "source": "coco",
  "question_type": "T1",
  "question": "Bu görselde kaç kişi bulunmaktadır?",
  "answer": "3",
  "language": "en"
}
```

**Alan açıklamaları:**

| Alan | Açıklama | Örnek Değerler |
|---|---|---|
| `id` | Benzersiz kayıt ID'si | `"coco_000001_T1"` |
| `image_id` | Kaynak dataset'teki orijinal ID | `"000001"` |
| `image_path` | `decisionmoe_dataset/` klasörüne göre relatif yol | `"images/coco/000001.jpg"` |
| `source` | Görüntü kaynağı | `"coco"`, `"ai2d"`, `"gqa"`, `"textvqa"` |
| `question_type` | Soru tipi | `"T1"`, `"T2"`, `"T3"`, `"T4"`, `"T5"` |
| `question` | Soru metni (İngilizce) | — |
| `answer` | Ground-truth cevap (İngilizce) | — |
| `language` | Dil | Her zaman `"en"` |

### 2.3 Split Oranları — Sabit

| Split | Oran | Kullanım |
|---|---|---|
| `train.json` | %70 | PPO training signal |
| `val.json` | %15 | Hyperparameter tuning |
| `test.json` | %15 | Final evaluation (dokunulmaz) |

Split **rastgele değil, stratified** olmalıdır: Her question type (T1-T5) ve her source (coco, ai2d, gqa, textvqa) her split'te orantılı temsil edilmeli.

### 2.4 Hedef Rakamlar — Sabit

| Metrik | Hedef |
|---|---|
| Toplam görüntü | ~2000 |
| Ham soru (filtresiz) | ~10.000 |
| Final soru (filtreli) | 7000–8000 |
| Her görüntüden soru sayısı | 5 (T1-T5, her tipten 1) |
| Her question type oranı | Yaklaşık eşit (%20 civarı) |

---

## 3. Görüntü Kaynakları

Dört açık kaynak dataset kullanılacak. Görüntüler indirilecek, annotation dosyaları da saklanacak.

### 3.1 COCO 2017 — Genel Sahneler

**Ne için:** T1 (visual perception), T2 (spatial reasoning), T4 (hallucination trap)

**İndir:**
- Görüntüler: [COCO 2017 Val Images](https://cocodataset.org/#download) (~1GB, 5000 görüntü)
- Annotation: `instances_val2017.json` (nesne kategorileri ve bbox)

**Kaç görüntü seçilecek:** 800 görüntü

**Seçim kriteri:** En az 3 farklı nesne kategorisi içeren görüntüler. Annotation dosyasından filtrelenebilir.

> ℹ️ COCO annotation dosyası T4 için kritiktir — "bu nesneler görselde yok" bilgisi buradan alınır, manuel kontrol gerekmez.

### 3.2 AI2D — Bilimsel Diyagramlar

**Ne için:** T3 (multi-hop QA), T5 (OCR + reasoning)

**İndir:**
- [AI2D Dataset](https://allenai.org/data/diagrams) (~1GB)
- Annotation JSON'ları ile birlikte gelir

**Kaç görüntü seçilecek:** 400 görüntü

**Seçim kriteri:** Hem metin hem görsel içeren diyagramlar. Sadece metin veya sadece görsel olanları dışarıda bırak.

### 3.3 GQA — Spatial Relation

**Ne için:** T2 (spatial reasoning) için en güçlü kaynak

**İndir:**
- [GQA Dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html)
- `val_balanced_questions.json` ile birlikte indir

**Kaç görüntü seçilecek:** 500 görüntü

**Seçim kriteri:** GQA'nın kendi soruları `spatial` kategorisinde olanlar öncelikli.

> ℹ️ GQA'nın kendi soru-cevapları doğrudan kullanılmaz — biz yeniden üretiyoruz. Yalnızca görüntüler alınır.

### 3.4 TextVQA — Metin İçeren Görseller

**Ne için:** T5 (OCR + reasoning)

**İndir:**
- [TextVQA Dataset](https://textvqa.org/dataset/)
- Val split görüntüleri yeterli

**Kaç görüntü seçilecek:** 300 görüntü

**Seçim kriteri:** Görüntüde okunabilir metin içeren sahneler (zaten tüm dataset bu amaçla toplanmış).

---

## 4. Soru Üretimi — Gemini 2.0 Flash

### 4.1 Model Seçimi

**Kullanılacak model:** Gemini 2.0 Flash  
**Neden:** Ücretsiz tier (günde 1500 istek), multimodal (görüntüyü doğrudan okur), JSON çıktısı güvenilir.

**API erişimi:** [Google AI Studio](https://aistudio.google.com/) — Google hesabıyla ücretsiz, kart istemiyor.

**Günlük limit:** 1500 istek → 2000 görüntü için 2 günde tamamlanır.

**Limit biterse:** [Groq API](https://console.groq.com/) üzerinden Llama-3.3-70B devreye girer (metin tabanlı, görüntü için önce caption üretmek gerekir).

### 4.2 Soru Üretim Promptu

Her görüntü için Gemini'ye gönderilecek prompt — **bu prompt değiştirilmemelidir:**

```
You are a dataset generation assistant for a multimodal AI research project.

Given the image, generate exactly 5 questions in the following types.
Each question must be answerable from the image alone.
Output must be valid JSON only — no explanation, no markdown, no preamble.

Question types:
- T1 (Visual Perception): Ask about something directly visible in the image.
  Example: "How many people are in this image?"
- T2 (Spatial Reasoning): Ask about the spatial relationship between two objects.
  Example: "Is the red car to the left or right of the bridge?"
- T3 (Multi-hop QA): Ask a two-step inference question using visual information.
  Example: "What color is the object on the table, and what does that color typically represent?"
- T4 (Hallucination Trap): Ask about something that is NOT in the image.
  The correct answer must be "no" or "none" or similar negation.
  Example: "Is there a dog in this image?" (only valid if there is no dog)
- T5 (OCR + Reasoning): Ask about text visible in the image and apply reasoning.
  Example: "Does the sign show a price above 10 dollars?"
  (Only generate T5 if the image contains readable text. If not, generate a second T3 instead.)

Output format (strict JSON array):
[
  {"question_type": "T1", "question": "...", "answer": "..."},
  {"question_type": "T2", "question": "...", "answer": "..."},
  {"question_type": "T3", "question": "...", "answer": "..."},
  {"question_type": "T4", "question": "...", "answer": "..."},
  {"question_type": "T5", "question": "...", "answer": "..."}
]
```

### 4.3 API Çağrısı — Python Referans Kodu

```python
import google.generativeai as genai
import json
import base64
from pathlib import Path

genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-2.0-flash")

PROMPT = """... (yukarıdaki prompt) ..."""

def generate_questions(image_path: str, image_id: str, source: str) -> list[dict]:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

    response = model.generate_content([
        {"mime_type": mime, "data": image_data},
        PROMPT
    ])

    raw = response.text.strip()
    questions = json.loads(raw)

    # ID ve metadata ekle
    results = []
    for q in questions:
        results.append({
            "id": f"{source}_{image_id}_{q['question_type']}",
            "image_id": image_id,
            "image_path": f"images/{source}/{Path(image_path).name}",
            "source": source,
            "question_type": q["question_type"],
            "question": q["question"],
            "answer": q["answer"],
            "language": "en"
        })
    return results
```

> ⚠️ API key asla kod içine yazılmaz — `.env` dosyasından `python-dotenv` ile okunur.

---

## 5. Kalite Filtresi — 4 Kural

Ham çıktı `raw/generated_raw.json`'a kaydedildikten sonra şu filtreler sırayla uygulanır:

### Kural 1 — Soru Uzunluğu

```python
5 <= len(question.split()) <= 40
```

5 kelimeden kısa ("What is this?") veya 40 kelimeden uzun sorular atılır.

### Kural 2 — T4 Doğrulaması (Sadece COCO için)

COCO annotation'larından görüntüdeki nesne kategorileri çekilir. T4 sorusunda sorulan nesne kategorisi gerçekten görselde **yoksa** geçer, **varsa** atılır.

```python
# COCO instances_val2017.json'dan yararlanılır
def validate_t4(image_id, question, coco_annotations):
    categories_in_image = get_categories(image_id, coco_annotations)
    questioned_object = extract_object_from_question(question)  # basit keyword match
    return questioned_object not in categories_in_image
```

AI2D, GQA, TextVQA görüntüleri için T4 soruları manuel gözden geçirilir (sample bazlı, tümü değil).

### Kural 3 — Cevap Çeşitliliği

T1, T2, T3 sorularının cevabı "yes", "no", "none" olamaz — bu tipler open-ended olmalı.

```python
FORBIDDEN_ANSWERS_FOR = {
    "T1": ["yes", "no", "none", "true", "false"],
    "T2": ["yes", "no", "none"],
    "T3": ["yes", "no", "none"],
}
```

### Kural 4 — Duplikasyon (Görüntü başına)

Aynı görüntüden üretilen 5 soru birbirine çok benziyorsa fazlası atılır.

```python
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model_st = SentenceTransformer("all-MiniLM-L6-v2")

def deduplicate(questions: list[str], threshold=0.85) -> list[str]:
    embeddings = model_st.encode(questions)
    # pairwise cosine similarity > threshold ise birini at
    ...
```

---

## 6. Stratified Split Üretimi

Filtreden sonra kalan ~7000-8000 soru train/val/test'e bölünür.

**Kural:** Her `(source, question_type)` kombinasyonu her split'te orantılı olmalı.

```python
from sklearn.model_selection import train_test_split

# Önce question_type + source bazlı grupla
# Sonra her gruptan %70/%15/%15 ayır
# Birleştir ve karıştır
```

**Çıktı:** `splits/train.json`, `splits/val.json`, `splits/test.json` — her biri `questions.json` ile aynı formatta.

---

## 7. Opsiyonel — Routing Oracle

Bu adım ana geliştirici ile koordineli yapılır. Atlanabilir ama paper için güçlü katkı sağlar.

**Ne yapılır:** Standart MoE-LLaVA-Phi2-2.7B modeli üzerinde her soru çalıştırılır ve hangi expert'lerin seçildiği loglanır.

**Çıktı formatı** (`routing_oracle.json`):

```json
{
  "coco_000001_T1": {
    "selected_experts": [0, 2],
    "expert_scores": [0.71, 0.33, 0.68, 0.42],
    "layer_activations": {"layer_8": [0, 2], "layer_16": [1, 2]}
  }
}
```

**Ana geliştirici bu dosyayı sağlayacak** — sen sadece formatı boş template olarak hazırla.

---

## 8. Haftalık Plan

| Hafta | Görev | Çıktı |
|---|---|---|
| **Hafta 1** | 4 dataset'i indir, görüntüleri seç ve `images/` klasörüne yerleştir | ~2000 görüntü hazır |
| **Hafta 1** | Gemini API erişimini kur, test et (10 görüntüyle deneme) | Çalışan üretim scripti |
| **Hafta 2** | Tüm görüntüler için soru üretimini çalıştır | `raw/generated_raw.json` |
| **Hafta 2** | 4 kalite filtresini uygula | `questions.json` (~7000-8000 kayıt) |
| **Hafta 3** | Stratified split üret | `splits/train.json`, `val.json`, `test.json` |
| **Hafta 3** | Ana geliştiriciye teslim + format doğrulama | ✅ Dataset hazır |

---

## 9. Claude Code Agent'a Verilecek Görev Özeti

Agent'a bu dokümanı verdikten sonra şu talimatları ekle:

```
Bu dokümandaki pipeline'ı uygula.

Kesinlikle değiştirme:
- Bölüm 2.1'deki dosya yapısı
- Bölüm 2.2'deki JSON formatı (alan adları dahil)
- Bölüm 2.3'teki split oranları (%70/%15/%15)
- Bölüm 4.2'deki soru üretim promptu

Serbest olduğun alanlar:
- Script mimarisi (hangi fonksiyonlar, hangi dosyalar)
- Hata yönetimi ve retry mekanizması
- Rate limit yönetimi (Gemini 1500/gün)
- Log formatı ve progress tracking
- Bölüm 3'teki görüntü indirme ve seçim detayları

Sırayla yap:
1. Önce klasör yapısını oluştur
2. Görüntü indirme scriptlerini yaz
3. Soru üretim scriptini yaz ve 10 görüntüyle test et
4. Kalite filtre scriptini yaz
5. Split scriptini yaz
6. Tümünü birleştiren bir main.py yaz
```

---

## 10. Teslim Kontrol Listesi

Ana geliştiriciye teslim öncesi şunları kontrol et:

- [ ] `decisionmoe_dataset/images/` altında 4 klasör mevcut
- [ ] `questions.json` geçerli JSON, her kayıt 8 alan içeriyor
- [ ] `id` alanları benzersiz (duplicate yok)
- [ ] `image_path` alanları gerçekten var olan dosyalara işaret ediyor
- [ ] `question_type` değerleri sadece T1-T5
- [ ] `language` alanı her kayıtta `"en"`
- [ ] `splits/` altında 3 dosya mevcut, toplamları `questions.json` ile eşleşiyor
- [ ] Her split'te T1-T5 dağılımı dengeli (±%5 tolerans)
- [ ] `raw/generated_raw.json` saklanmış (hata ayıklama için)

---

*Hazırlayan: DecisionMoE Araştırma Ekibi*
*Son güncelleme: Ağustos 2026*
