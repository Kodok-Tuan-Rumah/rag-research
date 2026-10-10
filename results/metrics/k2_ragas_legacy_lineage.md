# K2 — Lineage Run RAGAS Historis dan Mengapa Mereka Tidak Reproducible

Tanggal analisis : 2026-10-10
Berkas          : `results/metrics/ragas_results.json`, `ragas_phase2_postfix.json`,
                  `ragas_phase2_postfix_8b.json`
Status          : dokumentasi lineage. Tidak mengubah angka mana pun.

---

## 1. Koreksi atas Klaim Earlier

Saya sebelumnya menyatakan bahwa generation "tidak punya measurement coverage" dan
bahwa run berikutnya akan menjadi "pengukuran answer-quality pertama".

**Pernyataan itu salah.** Repo ini sudah memuat **9 run RAGAS** yang selesai
sebelum RAGAS kehilangan importability.

Yang benar adalah:

> Generation punya 9 run historis, tapi **tidak satu pun reproducible** di bawah
> konfigurasi canonical saat ini, dan **satu di antaranya korup**.

---

## 2. FACT — Inventaris Run

`results/metrics/ragas_results.json` berisi 7 run (JSON list, `save_result` menumpuk):

| # | exp_id | n | generator | judge | Faith | ARel | CPrec | CRec | Halluc |
|---|---|---|---|---|---|---|---|---|---|
| 1 | semantic_hybrid_rerank_v1 | 20 | — | — | 0.6754 | 0.9436 | 0.9500 | 0.8609 | 15.8% |
| 2 | semantic_hybrid_rerank_v1 | 20 | — | — | 0.6688 | 0.8112 | 0.8571 | 0.8571 | 12.5% |
| 3 | clean_dataset_v1 | 33 | llama-3.1-8b-instant | llama-3.1-8b-instant | 0.7987 | 0.7213 | 0.8445 | 0.9400 | 16.7% |
| 4 | clean_dataset_v2_70b_judge | 55 | llama-3.1-8b-instant | llama-3.3-70b-versatile | 0.8000 | 0.5118 | 0.7500 | 0.7500 | 20.0% |
| 5 | final_v1_70b_gen_8b_judge | 55 | llama-3.3-70b-versatile | llama-3.1-8b-instant | 0.7297 | 0.7438 | 0.8277 | 0.9313 | 18.4% |
| **6** | final_v1_70b_gen_8b_judge | 55 | openai/gpt-oss-20b | llama-3.1-8b-instant | **0.0** | **0.0** | **0.0** | **0.0** | **0.0%** |
| 7 | final_v1_70b_gen_8b_judge | 55 | openai/gpt-oss-20b | openai/gpt-oss-20b | 1.0000 | 0.4355 | 0.6861 | 0.8000 | 0.0% |

`ragas_phase2_postfix.json` (n=15) dan `ragas_phase2_postfix_8b.json` (n=15, n=28)
menambah 3 run lagi. Total **9 run**.

---

## 3. FACT — Run #6 korup, bukan hasil

```
faithfulness  : nan  (55 dari 55 sample)
answer_rel    : nan  (55 dari 55 sample)
context_prec  : nan  (55 dari 55 sample)
avg_*         : 0.0
hallucination : 0.0%
```

Seluruh job scoring gagal, RAGAS mengembalikan NaN (karena `raise_exceptions=False`
menelan error per-job), lalu `avg()` menyaring NaN dan mengembalikan `0.0`.

Jadi **run #6 bukan pengukuran**. Ia artifact dari kegagalan yang tercatat seolah-olah
hasil. Nilai `hallucination_rate: 0.0%` khususnya actively menyesatkan: ini berarti
"tidak ada sample yang fidelity < 0.5", yang pembaca baca sebagai "sangat baik",
padahal tidak ada satu pun sample yang dinilai.

Catatan tambahan: `json.dump` menulis `NaN` tanpa tanda kutip, yang **tidak valid
JSON ketat**. `json.load` Python menerimanya, konsumen JSON ketat akan menolak.

### Perbaikan

`src/evaluation/ragas_evaluator.py` kini:
- menghitung `nan_counts` per metrik dan menyimpannya di `AblationResult`,
- **melempar `RuntimeError`** bila ada metrik yang 100% NaN, alih-alih melaporkan 0.0,
- mencetak baris `[WARNING] incomplete metrics: ...` bila sebagian NaN.

Test: `tests/test_ragas_importable.py`.

---

## 4. FACT — Kenapa 9 run itu tidak reproducible

Dua hal independen, keduanya diverifikasi ulang pada 2026-10-10.

### 4.1 Dataset hilang

`find data -name "*clean*"` → kosong.

Run 1–7 memakai `clean_55_pairs` / `clean_33_pairs`, yang bersumber dari
`data/processed/ground_truth_qa.json`. Berkas itu **tidak ada lagi**. `run_eval.py`
lama masih hardcode path itu, jadi script tidak bisa dijalankan sama sekali.

Dataset canonical sekarang adalah `data/evaluation/retrieval_eval_dataset_v2.json`
(13 record, punya `ground_truth` terverifikasi sumber).

### 4.2 Model judge tidak bisa dipanggil

Diverifikasi langsung terhadap API, bukan dari ingatan:

```
llama-3.1-8b-instant      -> NotFoundError: 404 "model does not exist"
llama-3.3-70b-versatile   -> NotFoundError: 404 "model does not exist"
openai/gpt-oss-20b        -> OK
```

Run 3, 4, 5 memakai kedua model 404 tersebut. **6 dari 9 run tidak bisa diulang**
karena judge-nya sudah tidak ada.

---

## 5. INFERENCE

- Angka run 1–5 dan 7 **tidak bisa dikonfirmasi** di bawah konfigurasi canonical.
  Mereka memakai generator/judge dan dataset yang sudah tidak ada.
- Run 7 (semua 1.0 fidelity) adalah satu-satunya yang memakai model yang masih
  hidup (`openai/gpt-oss-20b`), tetapi tetap memakai dataset yang hilang — jadi
  tidak reproducible juga.
- Tidak ada run historis yang mengukur **pipeline canonical saat ini**
  (`MasterHybridRetriever` top_k=10 rerank=True + `GroqGenerator` +
  `retrieval_min_top1_score=0.0` + dataset v2).

## 6. UNKNOWN

- Apakah run 1–5 dan 7 dihasilkan dengan kode yang benar atau sudah sebagian rusak.
  `results/metrics/` tidak menyimpan log eksekusinya, dan working tree 2026-10-05
  tidak bisa direkonstruksi dari git (dicatat di `k1_baseline_freeze.md`).
- Berapa lama RAGAS sebenarnya functioning sebelum rusak. Tidak ada catatan.

---

## 7. Konsekuensi

1. Run RAGAS berikutnya pada dataset v2 adalah **run pertama yang reproducible**,
   bukan yang pertama secara historis.
2. Run #6 harus dibaca sebagai **FAILURE RECORD**, bukan "0.0". Dokumen ini tidak
   dihapus — menghapus bukti kegagalan bertentangan dengan disiplin evidence-first.
3. Angka lama **tidak boleh dibandingkan** dengan angka baru tanpa catatan
   eksplisit bahwa dataset, generator, dan judge semuanya berbeda.