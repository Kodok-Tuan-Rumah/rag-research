# K5 — Dua Perbaikan yang Mengubah Perilaku (Menunggu Keputusan)

Tanggal : 2026-10-11
Status  : **tidak dikerjakan.** Analisis dan proposal saja.
Alasan  : keduanya mengubah output retrieval/generasi pada jalur produksi.

Dua item lain di P3 sudah dikerjakan karena tidak mengubah perilaku:
balapan shared state (#13) dan threshold confidence (#12).

---

## 1. `reference_policy` tidak dapat dieksekusi

### FACT

`PolicyEngine.reference_policy()` mengembalikan `allow_references=True,
max_ref_ratio=0.4` untuk query bersinyal sitasi. Terukur pada
`results/metrics/k3_reference_impact.md`:

```
"Who wrote the Attention Is All You Need paper?"  factual  allow_ref=True  -> refs_in_final 0/10
"Who are the authors of the RAG paper?"           factual  allow_ref=True  -> refs_in_final 0/10
"What is the journal of Llama 2?"                 general  allow_ref=True  -> refs_in_final 0/10
"List the references cited by RAG-Token"          general  allow_ref=True  -> refs_in_final 0/10
```

Penyebabnya: `_section_boost` memberi 0.3 untuk section references, dan
kontribusi RRF adalah `w * boost / (k + rank + 1)`. Semua reference terbuang
di fusi, **tiga tahap** sebelum filter berjalan. Yang gagal adalah kebijakannya
— ia memberi diri sendiri wewenang mengizinkan sesuatu yang sudah dibuang.

###ineteksisting_options

| Opsi | Perubahan | Risiko |
|---|---|---|
**A** | `rrf_fuse` menerima **boost per-query**. `_section_boost` menerima flag `allow_references`; kalau `True`, references mendapat 1.0 | Retrieval berubah **hanya** untuk query sitasi. Perlu baseline baru untuk subset itu. `tests/test_section_boost.py` harus diperluas |
**B** | Naikkan boost references dari 0.3 ke mis. 1.0 secara global | **Tidak Doing.** Meng prepping semua query non-sitasi untuk Answers about references |
**C** | Biarkan; catat sebagai keterbatasan yang diketahui | Nol biaya; kemampuan "who wrote X?" tetap tidak ada |

### Rekomendasi: A

Satu-satunya yang membuat kebijakan sesuai janji tanpa mengganggu query lain.
Perlu keputusan eksplisit karena ia mengubah output retrieval.

---

## 2. A5 — retry API tidak sama dengan Agent

### FACT

§3.4 menyebut tiga aksi saat retry. Terukur mana yang benar-benar berfungsi:

| Aksi §3.4 | Agent | API | Efektif? |
|---|---|---|---|
Lebarkan `k` ×1.5 | `agent.py:151` | **tidak ada** | Agent ya |
Geser bobot BM25 | `agent.py:152-153` | `main.py:228` | keduanya ya |
Longgarkan filter | `agent.py:154` (−0.1) | **tidak ada** | **keduanya tidak** |

Aksi ketiga tidak efektif di **kedua** implementasi. Alasannya terukur: skor RRF
berada di rentang **[0.005, 0.021]** (120 sampel, 3 query). Floor `0.0` dan
relaksasi `−0.1` sama-sama membuang **0 chunk**.

Jadi koreksi terhadap klaim saya sebelumnya: Agent bukan 3/3, melainkan **2/3**.

### Dua celah yang tersisa, danbedanya

| Perilaku | Agent | API |
|---|---|---|
Pemicu retry | `REJECT` **dan** `unstable_retrieval` | `REJECT` saja |
Stagnation guard | ada (`agent.py:212-220`) | tidak ada |

### Opsi

| Opsi | Perubahan | Risiko |
|---|---|---|
**A** | Samakan: API retry juga saat `unstable_retrieval`, dengan stagnation guard | **Perilaku API berubah.** Query dengan retrieval tidak stabil akan retry — menambah latency |
**B** | Dokumentasikan perbedaan; jangan samakan | Nol risiko, tapi perilaku tetap berbeda antar adapter |
**C** | Angkat `unstable_retrieval` kegate CI | Gate baru yang mungkin langsung merah |

### Rekomendasi: B, lalu A setelah §3.4 diperbaiki

**Alasan:** yang/actions §3.4 tidak direvisi, dan "menyamakan" API ke Agent
hanya akan menyalin spesifikasi yang cacat — termasuk relaksasi filter yang
tidak bekerja. Yang perlu pertama: **perbaiki spesifikasi §3.4** agar aksi ketiga
punya efek nyata (lihat K3 / Appendix B), baru samakan adapter.

Menyalin perilaku yang salah ke adapter kedua tidak memperbaiki apa pun; ia
menggandakan kesalahan.

---

## Yang Sudah Dikerjakan di P3 (tanpa perubahan perilaku)

| Item | Hasil |
|---|---|
Balapan shared state | `_run_search` tidak lagi menulis `self.use_multi_query` / `self.use_hyde`. Flag jadi parameter `_expand_and_hyde_parallel`. Parameter konstruktor yang selalu ditimpa dihapus. `tests/test_no_shared_request_state.py`, termasuk uji dua thread dengan flag berlawanan |
Threshold confidence | `0.35` / `0.65` kini dari `Settings`. Diverifikasi **0 selisih dari 2.000 titik × 6 kombinasi threshold** terhadap implementasi lama. `tests/test_confidence_thresholds.py` |

Untuk threshold ada satu hal yang perlu dicatat karena mungkin tidak disengaja:
`reject` di-clamp ke `0.0` untuk tipe `reasoning`. Kalau `reject_threshold`
di-set di bawah `0.05`, query reasoning **tidak bisa ditolak sama sekali**.
Perilaku itu **sudah ada sebelumnya** dan sengaja dipertahankan, bukan bug baru.

---

*Analisis mengikuti §14: ukur dulu, jangan mengerjakan perubahan perilaku tanpa
persetujuan eksplisit.*