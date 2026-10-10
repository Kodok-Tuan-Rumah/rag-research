# K4 — Apa yang Membuat Proyek Ini Over-Engineering

Tanggal ukur : 2026-10-11
Status        : analisis penyebab + perbaikan yang sudah diterapkan
Artefak       : `scripts/MANIFEST.json`, `scripts/check_script_manifest.py`

---

## 1. FACT — Peta kuantitas

| Lapisan | File | LOC | Rasio |
|---|---:|---:|---|
**`src/`** | 23 | **5.281** | — |
`tests/` | 29 | 4.735 | test:src = 0.90 |
**`scripts/`** | 39 | **8.453** | **scripts:src = 1.60** |
`archive/` | 13 | 2.490 | sudah dipensiunkan |
`results/metrics/` | 124 file | — | untuk dataset **14 query** |

```
kode mati di src/     : 13 dari 180 fungsi  (7.2%)
duplikasi identik    : 3 kasus lintas layer
script tak terdokumentasi : 29 dari 38
```

---

## 2. FACT — `src/` sendiri tidak berlebihan

5.281 baris mencakup ingestion, embedding, BM25, RRF, cross-encoder rerank,
chunk filtering, policy engine, confidence engine, abstention, retry
orchestration, LLM generation, FastAPI, dan RAGAS. Sekitar **230 baris per
komponen**.

Kode mati 7.2% dan duplikasi identik hanya 3 kasus. Arsitekturnya benar-benar
canonical: satu `MasterHybridRetriever`, API/CLI/Agent hanya adapter.

**Over-engineering tidak ada di produk.**

---

## 3. INFERENCE — Lima penyebab, bukan gejalanya

### Penyebab 1 — Tidak ada yang pernah menjalankan script

248 test tidak mengeksekusi satu pun script. Setiap script ditulis, dijalankan
sekali, buktinya diambil, lalu tidak pernah ditinjau ulang. Tidak ada mekanisme
yang bisa mengatakan "ini sudah mati".

Bukti: 29 dari 38 script tidak disebut di README, CI, atau Makefile mana pun.

### Penyebab 2 — Artefak jadi tempat tenggelam

`results/metrics/` berisi 124 file untuk dataset 14 query. Menulis diberi
imbalan; memangkas tidak pernah. `fase_f_reranker_ablation.json` punya kembaran
`.replica_20261010.json` — jejak re-run yang dilakukan ad-hoc.

### Penyebab 3 — Dataset dihapus, konsumennya tidak ikut disapu

Migrasi ke `data/evaluation/` memindahkan file tapi tidak menyapu siapa yang
membacanya. 8 script langsung mati tanpa satu pun alarm.

### Penyebab 4 — Tidak ada registry entrypoint

Empat skrip RAGAS paralel. Dua implementasi chunking penuh. Tiga skema `top_k`.
Aturan top-3 `PARTIAL_TRUST` diempat tempat. Semuanya lahir karena tidak ada
pertanyaan "apakah ini sudah ada?" sebelum yang berikutnya ditulis.

### Penyebab 5 — Identitas git placeholder

21 commit tercatat atas nama `Your Name <you@example.com>`. `git log
--author=` tidak berguna, sehingga "apa yang benar-benar dipakai" mustahil
ditelusuri secara retrospektif. Ini memblokir arkeologi.

---

## 4. Yang diperbaiki

| Perubahan | Efek |
|---|---|
`scripts/MANIFEST.json` + `check_script_manifest.py` | Setiap script wajib punya status; CI gagal kalau tidak |
`status` library | `run_eval_phase2.py` yang tampak mati dinyatakan hidup — `token_f1` di dalamnya adalah metrik Semantic F1 §2.1 |
`ACKNOWLEDGED_DEAD_PATHS` | Kepatuhan yang disengaja dicatat beserta alasannya, bukan ditoleransi diam-diam |
`coverage_metric.py` default diarahkan ke `data/evaluation/` | Path default yang rusak diperbaiki |
README "Script inventory" | 3 script canonical/library kini terdokumentasi |

### Koreksi atas rekomendasi saya sendiri

Saya mengusulkan mengarsipkan `coverage_metric.py` (767 baris),
`audit_k710_coverage.py` (342), `audit_k712_q11_q20.py` (368), dan
`run_k711_latency_replication.py` (467) — total 1.944 baris.

**Empatnya ternyata tidak mati.** Heuristik yang saya pakai salah: `("=" in
line)` menandai setiap konstanta sebagai penulisan, sehingga `GT_FILE = ...`
disalahartikan sebagai output. Setelah penelusuran AST yang benar, hanya satu
script yang benar-benar membaca data hilang.

Saya juga sempat mengarsipkan `run_eval_phase2.py`, lalu mendapati
`tests/test_em_normalization.py` mengimpornya — dan itu menemukan bahwa
`token_f1` sudah ada di sana. Dikembalikan.

Dua dari empat koreksi ini adalah kesalahan saya sendiri, ditemukan karena
saya memeriksa buktinya alih-alih menerima output alat.

---

## 5. Yang tersisa — di luar lingkup overhaul

| Temuan | Ukuran | Status |
|---|---|---|
`scripts/` masih 1.6× `src/` | ~6.500 baris setelah arsip | **tercatat**, belum dipangkas |
124 artefak `results/metrics/` | tanpa index | belum diindeks |
Empat implementasi top-3 rule | semantik | belum |
Tiga skema `top_k` | semantik | belum |
Dua implementasi chunking | 582 vs 796 chunk | `chunker.py` sudah dipisah, belum diarsipkan |
13 fungsi mati di `src/` | 7.2% | belum |
Identitas git placeholder | 21 commit | **butuh Anda** |

---

## 6. Verifikasi

| Cek | Hasil |
|---|---|
`check_script_manifest.py` | exit 0, 37 script terklasifikasi |
`check_ci_contracts.py` | exit 0, 4/4 |

Tidak ada perubahan pada `src/retrieval`, `src/generation`, atau korpus, jadi
baseline retrieval tidak dapat bergerak — tetap diverifikasi per §14.

---

*Analisis mengikuti disiplin §14: ukur dulu, jangan menerima output alat tanpa
memeriksa prasyaratnya.*