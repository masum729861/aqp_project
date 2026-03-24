
📘 README.md (Your Project)
Adaptive Sampling for Fast Approximate Query Processing in Large Databases
📌 Overview

This project implements an Approximate Query Processing (AQP) framework using Adaptive Sampling to efficiently estimate query results on large datasets.

Instead of scanning the entire dataset (as in traditional exact queries), this system dynamically samples data and stops early when the result is accurate enough, significantly reducing computation while maintaining low error.

The project compares four approaches:

Exact Query Processing
Uniform Sampling
Stratified Sampling
Adaptive Sampling (proposed approach)


🚀 Key Features
✅ Adaptive sampling with dynamic stopping condition
✅ Confidence interval–based error estimation
✅ Comparison across multiple sampling strategies
✅ Support for common aggregate queries:
COUNT
SUM
AVG
✅ Experimental evaluation with:
Multiple trials
Dataset scaling
Error and time analysis
✅ Automatic generation of:
Tables (CSV)
Performance graphs


🧠 How It Works
Adaptive Sampling Logic

The algorithm:

Starts with a small sample of the dataset
Computes an estimate (e.g., average)
Calculates error using confidence intervals
Checks if error is below a threshold
If yes → stops early
If no → increases sample size and repeats

👉 This avoids processing unnecessary data.



📊 Experimental Setup
Dataset size: 10,000 rows
Number of trials: 10
Sampling strategies compared:
Uniform Sampling
Stratified Sampling
Adaptive Sampling
Metrics evaluated:
Execution Time
Relative Error
Confidence Intervals


📈 Results Summary
Adaptive sampling achieved low error (~3%)
The algorithm stopped early using only 5% of the dataset
Demonstrates that:
High accuracy can be achieved with limited data
Full dataset scanning is often unnecessary


📂 Project Structure
aqp_project/
│
├── dataset/
│   └── sales.csv
│
├── src/
│   ├── exact_query.py
│   ├── uniform_sampling.py
│   ├── stratified_sampling.py
│   ├── adaptive_sampling.py
│   ├── run_experiments.py
│   └── generate_big_sales_dataset.py
│
├── results/
│   ├── tables/
│   └── figures/
│
└── README.md
