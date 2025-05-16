# Pseudo-spectre Project

This project is developed as part of a course or research project related to high-performance computing and algorithm comparison. It explores and benchmarks two different numerical approaches — **GRID-based** and **Newton-based prediction-correction** — along with their parallel implementations and performance evaluations.


## Project Structure

| File | Description |
|------|-------------|
| `GRID.py` | Implementation of the baseline algorithm using a grid-based discretization method. |
| `Prediction-correction_Newton.py` | Implementation of the prediction-correction algorithm using Newton's method. |
| `version_superposees.py` | A hybrid or combined version of the GRID and Newton approaches. |
| `comparaison_algo.py` | Compares the performance of GRID and Newton methods based on matrix size. |
| `benchmark.py` | Parallelized versions of GRID and Newton; measures execution time depending on the number of CPU cores. |
| `bench_speedup.py` | Builds on `benchmark.py` by also computing the **speedup** obtained through parallelization. |
| `Sujet Projet CCA.pdf` | Original project statement or assignment document (in French). Provides context and objectives. |


## Usage

You can run each script individually depending on what analysis you want to perform. For example:

python3 comparaison_algo.py
python3 benchmark.py
python3 bench_speedup.py


Make sure to have Python 3 and required libraries (e.g. NumPy, time, multiprocessing) installed.


