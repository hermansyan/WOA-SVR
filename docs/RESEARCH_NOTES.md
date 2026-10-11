# Catatan Riset — WOA-SVR

## Ringkasan

Repo ini mengimplementasikan **Support Vector Regression (SVR)** yang
hyperparameter-nya (`cLR`, `C`, `epsilon`, `lamda`, `sigma`) dioptimasi
dengan **Whale Optimization Algorithm (WOA)**. SVR ditulis dari nol
(kernel RBF + sequential learning, tanpa scikit-learn) sebagai bahan
pembelajaran dan eksperimen.

## Log perbaikan bug (2026-10-11)

Perbaikan dilakukan terhadap implementasi WOA dengan merujuk pada
formulasi kanonik Mirjalili & Lewis (2016):

| # | Lokasi lama | Masalah | Perbaikan |
|---|---|---|---|
| 1 | `main.py`, `main_svr.py` | Parameter `a` *naik* 0→2 (`a = ((t+1)/T)*2`), padahal paper menetapkan `a` *turun* 2→0 — jadwal eksplorasi/eksploitasi terbalik | `woa.decreasing_a()`: `a = 2 − ((t+1)/T)*2` |
| 2 | `main.py`, `main_svr.py` | `int(random()*len(solutions))` memakai jumlah *parameter* (2/5) sebagai jumlah *agen* (40/10) setelah transpose → fase exploration lumpuh | `rng.integers(0, n_agents)` — seleksi atas populasi yang benar |
| 3 | `main_svr.py::validate_range_solution` | Clamp `lamda` memakai `UPPER_BOUND_EPSILON` (0.1) bukan `UPPER_BOUND_LAMBDA` (0.5) → separuh ruang pencarian λ tak terjangkau | Clamp generik `clip_to_bounds()` per dimensi di `woa.py` |
| 4 | `training_svr.py` | `print(exec_svr_woa())` + `result = exec_svr_woa()` di level modul → training berjalan 2× setiap import | Seluruh eksekusi dipindah ke `if __name__ == "__main__":` |
| 5 | `training_svr.py::normalization` | min/max dihitung dari *seluruh* data (termasuk test) → **data leakage** | Statistik hanya dari porsi training; test memakai statistik yang sama |
| 6 | `main.py` | Label plot `"GA Result"` pada kode WOA (sisa copy-paste) | `"WOA Result"` |

Perubahan struktural:

- Logika WOA diekstrak ke modul generik `woa.py` (bisa dipakai ulang untuk
  fungsi objektif apapun) — `main.py`/`main_svr.py` menjadi driver tipis.
- `training_svr.py` di-refactor menjadi `svr.py` (API bersih, tanpa
  eksekusi saat import); file lama dihapus (riwayat tersimpan di git).
- Dead code dihapus: blok MySQL yang dikomentari, `decimal_normalization`
  (spesifik komoditas, tidak dipakai), variabel global bayangan.
- `exec_svr_woa` kini juga menghitung prediksi & MSE *test* yang jujur
  (`model.test_prediction`, `model.test_mse`) — tidak dipakai untuk
  optimasi, hanya dilaporkan.

## Keterbatasan yang diketahui

1. **Fitness = MSE training.** WOA mengoptimasi hyperparameter terhadap
   error training → risiko overfitting. Untuk riset serius, gunakan
   MSE validasi / cross-validation sebagai fitness.
2. **Dataset kecil** (21 baris). Hasil bersifat demonstratif.
3. **Performa**: loop Python O(n²) untuk matriks jarak & kernel.
   Vektorisasi NumPy akan mempercepat orde besaran.
4. Belum ada *early stopping* berbasis validasi; kriteria berhenti =
   konvergensi delta-α atau `iter_max`.

## Roadmap yang disarankan

- [ ] Fitness berbasis validasi (atau k-fold CV) untuk optimasi yang jujur
- [ ] Vektorisasi komputasi kernel dengan NumPy
- [ ] Perbandingan baseline: `sklearn.svm.SVR` + `GridSearchCV` pada data sama
- [ ] Dataset riil yang lebih besar + dokumentasi sumber data
- [ ] Eksperimen reproduksibel: simpan hasil tiap run (CSV/JSON), bukan
      hanya plot
