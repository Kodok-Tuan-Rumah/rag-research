# K3 — Dampak 258 Chunk REFERENCES pada Pipeline

Tanggal ukur : 2026-10-10
Artefak      : `results/metrics/k3_reference_impact.json`
Skrip        : `scripts/measure_reference_impact.py` (dapat dijalankan ulang)
Dasar        : 13 query canonical `retrieval_eval_dataset_v2.json`, `top_k=10`, `rerank=True`

Status: **pengukuran, bukan perbaikan.**

---

## 1. FACT — Komposisi korpus

```
total chunk                : 796
chunk bertanda REFERENCES  : 258  (32.4%)
porsi karakter             : 30.7%
median panjang teks refs    : 411 karakter
median panjang teks non-ref: 462 karakter
length_separable           : False   (distribusi sangat bertumpuk)
```

Referensi bukan chunk pendek yang mudah dibuang — texts-nya hampir sama
panjang dengan konten. Jadi tidak bisa dipisahkan dengan ambang panjang;
hanya metadata section yang bisa.

---

## 2. FACT — Tidak ada label gold yang menunjuk REFERENCES

| Dataset | Query | Gold → REFERENCES |
|---|---:|---:|
`retrieval_eval_dataset.json` | 14 | **0** |
`retrieval_eval_dataset_v2.json` | 13 | **0** |
`retrieval_eval_dataset_v3.json` | 10 | **0** |
`retrieval_train_eval.json` | 8 | **0** |
`retrieval_holdout_eval.json` | 6 | **0** |

Ini yang membuat keputusan mudah: membuang references dari korpus
tidak akan membuat benchmark mustahil.

---

## 3. FACT — Bocor per tahap

13 query canonical, rerank aktif:

| Tahap | References | Total | Bocor |
|---|---:|---:|---:|
`dense` | 93 | 520 | **17.9%** |
`bm25` | 87 | 520 | **16.7%** |
`rrf` | **0** | 520 | **0.0%** |
`reranked` | 0 | 260 | **0.0%** |
`final` | 0 | 130 | **0.0%** |

Dense dan BM25 **berhasil mengambil** chunk references — cukup banyak. Tapi
fusi RRF membuang **seluruhnya**.

## 4. FACT — Penyebabnya section boost

`hybrid_retriever.py:177-178` memberi boost **0.3** untuk section
references/bibliography, sedangkan section method mendapat 1.4. Kontribusi RRF
adalah `w * boost / (k + rank + 1)` (`hybrid_retriever.py:59`), jadi boost 0.3
membagi kontribusi ~3.3× dan cukup untuk mendorong seluruh references keluar
dari 40 teratas.

Evidence tambahan:

```
input reranker : 0/260  (0.0%)
diblokir filter strict : 0 kemunculan
```

Filter strict **tidak pernah menyala**. Section boost melakukan seluruh
pekerjaan, di hulu.

---

## 5. FACT — Yang tersisa hanyalah biaya index-time

| Aspek | Nilai |
|---|---|
Vektor dense untuk references | ~387 KB dari total ~1.19 MB |
Ukuran index Qdrant di disk | 3.8 MB (collection) |
Waktu embedding saat ingestion | dibagi 2 terhadap korpus 538 chunk |
Waktu query | **nol perubahan terukur** |

Ingestion berjalan sekali, bukan per query.

---

## 6. INFERENCE

Tidak ada alasan untuk membuang references dari korpus.

- Biaya: ~387 KB dan sebagian waktu ingestion **sekali jalan**.
- Menghapus berarti **re-ingest + re-index**. `data/evaluation/`
  `retrieval_eval_dataset_v2.json` menyimpan `corpus_chunks_sha256` di envelope —
  mengubah korpus membatalkan sidik jari itu dan memaksa seluruh baseline
  diulang.
- Manfaat: nol perbaikan query-time terukur.

---

## 7. Koreksi atas dugaan saya sendiri

Saya menulis sebelum mengukur:

> *"32% kandidat yang harus di-rerank adalah teks bibliografi. Beban yang tidak perlu."*

**Salah.** Terukur: **0 dari 260** input reranker adalah chunk references.
Tidak ada beban terbuang sama sekali.

---

## 8. FACT — `reference_policy` ternyata tidak dapat dieksekusi

Diuji langsung dengan query yang secara eksplisit **meminta** informasi
referensi:

| Query | Tipe | `allow_references` | `max_ref_ratio` | References di `final_ids` |
|---|---|---|---|---:|
"Who wrote the Attention Is All You Need paper?" | factual | **True** | 0.4 | **0/10** |
"Who are the authors of the RAG paper?" | factual | **True** | 0.4 | **0/10** |
"What is the journal of Llama 2?" | general | **True** | 0.4 | **0/10** |
"List the references cited by RAG-Token" | general | **True** | 0.4 | **0/10** |
"What is attention mechanism in transformer models?" | general | False | 0.0 | 0/10 |
"How does hybrid retrieval work?" | reasoning | False | 0.0 | 0/10 |

Kebijakan mengizinkan referensi untuk empat query pertama. Hasilnya tetap nol,
karena section boost sudah menghapus semuanya di fusi RRF.

**Kesimpulan: sistem secara struktural tidak dapat menjawab pertanyaan tentang
referensi**, meski `PolicyEngine.reference_policy()`romise bahwa ia boleh.

Ini berbeda dari "filter tidak bekerja" — filter memang bekerja, hanya tidak
punya benda untuk difilter. Yang tidak bekerja adalah **kebijakan**: ia
memberi diri sendiri wewenang untuk mengizinkan sesuatu yang sudah dibuang
tiga tahap sebelumnya.

Tidak ada filter yang bisa memperbaiki ini. Yang diperlukan adalah menghormati
section boost untuk query sitasi, yaitu **mengubah `rrf_fuse` agar menerima
boost per-query**, bukan boost global. Itu perubahan perilaku retrieval dan
**di luar lingkup pengukuran ini** — dicatat sebagai temuan, bukan dikerjakan.

---

## 9. Ringkasan

| Temuan | Tindakan |
|---|---|
32.4% corpus adalah references | **tidak ada tindakan** — nol biaya query-time |
Section boost menahan 100% kebocoran | **tidak ada tindakan** — sudah bekerja |
`reference_policy` tidak dapat dieksekusi | **catatan untuk keputusan Anda** — butuh perubahan perilaku retrieval |

Yang direkomendasikan hanya satu: **catat hasil ini**, supaya rasanya
seperti masalah besar di audit berikutnya — padahal ia bukan masalah, dan
masalah kedua (`reference_policy`) tersembunyi tepat di balik solusi yang
berhasil.

---

*Measured per §14 discipline: artifact regenerated by
`scripts/measure_reference_impact.py`, no LLM, no network.*