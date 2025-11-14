\# Code for Numerical Experiments in "Prehospital Transport Strategy Selection in Regional Stroke Networks"



This repository contains the Python code used to generate the numerical results and figures in the paper:



> Prehospital Transport Strategy Selection in Regional Stroke Networks: A Spatial Point Process Perspective



The code implements three numerical experiments:



1\. Numerical evaluation of the function \\(\\kappa(\\rho)\\) from its integral definition.

2\. Monte Carlo (PPP) validation of the optimal strategy boundary \\(\\theta^\*(\\rho)\\) and the dimensionless distance difference \\(\\Delta(\\theta,\\rho)\\).

3\. Finite-size analysis in binomial point process (BPP) models, including boundary deviations, asymptotics, and \\(\\kappa\\)-related corrections.



---



\## Repository structure



\- `kappa\_rho\_integral.py`  

&nbsp; \*\*Experiment 1 – Integral-based evaluation of \\(\\kappa(\\rho)\\)\*\*  

&nbsp; Computes and plots \\(\\kappa(\\rho)\\) for a range of \\(\\rho = \\lambda\_p / \\lambda\_c\\) using:

&nbsp; - adaptive Simpson integration on \\(\[0, 2]\\), and  

&nbsp; - a closed-form tail integral on \\((2, \\infty)\\).  

&nbsp; The script:

&nbsp; - generates a curve \\(\\kappa(\\rho)\\) on a user-specified grid of \\(\\rho\\),

&nbsp; - saves the curve as a PNG figure,

&nbsp; - exports sampled \\((\\rho, \\kappa(\\rho))\\) pairs to a CSV file,

&nbsp; - prints representative values of \\(\\kappa(\\rho)\\) at selected \\(\\rho\\) values.



\- `Boundary\_Characterization.py`  

&nbsp; \*\*Experiment 2 – PPP boundary characterization and \\(\\Delta(\\theta,\\rho)\\)\*\*  

&nbsp; Monte Carlo validation of the optimal strategy boundary \\(\\theta^\*(\\rho)\\) in a planar Poisson point process (PPP) model.  

&nbsp; The script:

&nbsp; - simulates PPPs for CSCs and PSCs on a torus of side length \\(L\\),

&nbsp; - computes Monte Carlo estimates of \\(\\alpha(\\rho)\\) and \\(b(\\rho)\\) (in a dimensionless, scaled form),

&nbsp; - estimates \\(\\theta^\*(\\rho) = \\alpha(\\rho)/b(\\rho)\\) with standard errors and 95% confidence intervals,

&nbsp; - constructs \\(\\Delta(\\theta,\\rho) = \\alpha(\\rho) - \\theta\\, b(\\rho)\\),

&nbsp; - generates heatmaps, \\(\\Delta\\)–\\(\\theta\\) slices, and boundary alignment plots,

&nbsp; - saves CSV files with main estimates and sanity-check statistics.



\- `bpp\_experiment.py`  

&nbsp; \*\*Experiment 3 – BPP vs PPP finite-size analysis\*\*  

&nbsp; Numerical experiment for binomial point processes (BPP) with a finite number of CSCs (\\(N\_c\\)) and PSCs (\\(N\_p\\)).  

&nbsp; The script:

&nbsp; - simulates BPP realizations for multiple \\((N\_c, N\_p)\\) configurations on a torus,

&nbsp; - estimates \\(\\theta^\*(\\rho)\\) under BPP and compares it to PPP theory,

&nbsp; - computes dimensionless quantities \\(\\tilde{D}\_c\\), \\(\\tilde{D}\_{\\min}\\) and their PPP baselines,

&nbsp; - analyzes finite-size deviations of the boundary and distance metrics,

&nbsp; - performs through-origin regression for \\(1 - \\tilde{D}\_{\\min}\\) vs \\(1/N\\),

&nbsp; - studies the collapse of \\(N (1 - \\tilde{D}\_{\\min})\\) and the relative deviation of \\(\\kappa\\),

&nbsp; - produces a set of figures (F1–F4) corresponding to the BPP analysis.



\- `README.md`  

&nbsp; This file.



\- `requirements.txt`  

&nbsp; List of Python package dependencies needed to run all experiments.



---



\## Environment



The code has been tested with:



\- Python 3.8 or higher (e.g., Python 3.10)

\- NumPy

\- SciPy

\- Matplotlib

\- pandas

\- mpmath



The exact dependencies (with minimal version requirements) are listed in `requirements.txt`.



---



\## Installation



It is recommended to use a virtual environment:



```bash

python -m venv venv





\# On Windows:

\# venv\\Scripts\\activate



pip install -r requirements.txt



