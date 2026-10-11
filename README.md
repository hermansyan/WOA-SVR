# WOA-SVR

Optimasi hyperparameter **Support Vector Regression (SVR)** menggunakan
**Whale Optimization Algorithm (WOA)** — algoritma metaheuristik yang meniru
perilaku berburu paus bungkuk (*bubble-net feeding*).

Metode WOA mengikuti formulasi asli Mirjalili & Lewis (2016):
tiga mekanisme update posisi — *encircling prey*, *spiral bubble-net*,
dan *search for prey* — dengan parameter kontrol `a` yang turun linear
dari 2 ke 0 untuk menyeimbangkan eksplorasi dan eksploitasi.

> **Catatan riset**: repo ini adalah studi SVR+WOA. Riset terkait penulis
> yang memakai pasangan metode serupa (ELM+WOA untuk peramalan Covid-19)
> dipublikasikan di Jurnal J-PTIIK — lihat [docs/REFERENCES.md](docs/REFERENCES.md).

## Struktur proyek

```
WOA-SVR/
├── woa.py            # Implementasi WOA kanonik (optimizer generik)
├── svr.py            # Mesin SVR: kernel RBF + sequential learning (tanpa sklearn)
├── SVR_Model.py      # Kontainer hyperparameter & hasil training
├── main_svr.py       # Eksperimen utama: WOA mencari 5 hyperparameter SVR
├── main.py           # Demo: WOA untuk regresi linier y = m*x + c
├── data/
│   ├── dataCovid.csv   # Data deret waktu (format: fitur..., target)
│   └── regression.xlsx # Data demo regresi linier
├── tests/            # Unit test (pytest)
└── docs/
    ├── REFERENCES.md      # Referensi jurnal & paper
    └── RESEARCH_NOTES.md  # Catatan riset, bug yang diperbaiki, roadmap
```

## Instalasi

```bash
pip install -r requirements.txt
```

## Penggunaan

```bash
# Eksperimen utama: optimasi 5 hyperparameter SVR dengan WOA
python main_svr.py --agents 10 --iterations 10 --seed 42

# Demo WOA pada regresi linier
python main.py --agents 40 --iterations 20 --seed 42

# Menjalankan test
python -m pytest tests/ -v
```

## Metode singkat

1. **WOA** (`woa.py`): populasi paus diinisialisasi acak dalam batas
   pencarian; tiap iterasi posisi diupdate lewat mekanisme *encircling*
   (`|A| < 1`), *search* (agen acak, `|A| >= 1`), atau *spiral* (`p >= 0.5`).
2. **SVR** (`svr.py`): normalisasi min-max, matriks kernel RBF dari jarak
   antar data, matriks Hessian (`K + λ²`), lalu *sequential learning*
   untuk variabel dual `α / α*`.
3. **WOA-SVR** (`main_svr.py`): tiap paus = vektor
   `[cLR, C, epsilon, lamda, sigma]`; fitness = MSE training SVR.

## Keterbatasan (jujur)

- Dataset bawaan kecil (21 baris) — cocok untuk demo, bukan untuk
  klaim generalisasi.
- Fitness optimasi memakai MSE *training*; MSE *test* dilaporkan terpisah
  sebagai pembanding yang jujur.
- Implementasi memakai loop Python murni (O(n²)) — lambat untuk data besar;
  vektorisasi NumPy adalah pekerjaan rumah yang bagus.

Detail lengkap ada di [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md).

## Referensi

- Mirjalili, S. & Lewis, A. (2016). The Whale Optimization Algorithm.
  *Advances in Engineering Software*, 95, 51–67.
- Li, L. et al. (2019). Extreme learning machine optimized by whale
  optimization algorithm using insulated gate bipolar transistor module
  aging degree evaluation. *Expert Systems with Applications*, 127, 58–67.
- Syantoso, H. & Soebroto, A. A. Peramalan Kasus Positif Harian Covid-19
  di Indonesia Menggunakan Metode ELM dengan Optimasi WOA.
  *Jurnal J-PTIIK, Universitas Brawijaya*.

Lihat [docs/REFERENCES.md](docs/REFERENCES.md) untuk daftar lengkap.
