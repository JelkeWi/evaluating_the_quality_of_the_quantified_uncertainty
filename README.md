# Evaluating the Quality of the Quantified Uncertainty for (Re)Calibration of Data-Driven Regression Models
This repository contains the code to reproduce the benchmark of our paper [Evaluating the Quality of the Quantified Uncertainty for (Re)Calibration of Data-Driven Regression Models](https://doi.org/10.1016/j.ijar.2026.109685).


* [benchmark_real_synthetic.py](benchmark_real_synthetic.py) is used to run the first three case studies of the benchmark.
  * run `python benchmark_real_synthetic.py --create_synthetic_data --create_real_data --run_real_data --run_synthetic_data --recalibration --n_runs 1 --name synthetic_and_real1`
* [benchmark_miscalibration.py](benchmark_miscalibration.py) is used to run the controlled miscalibration case study.
  * run `python benchmark_miscalibration.py --create_real_data --run_real_data --n_runs 100 --name miscalibration1`

Once the benchmark results are available, they can be visualized. A Python notebook is available for each benchmark part.

Notebooks:
* [CaseStudy_real.ipynb](CaseStudy_real.ipynb)
* [CaseStudy_synthetic.ipynb](CaseStudy_synthetic.ipynb)
* [CaseStudy_recalibration.ipynb](CaseStudy_recalibration.ipynb)
* [CaseStudy_miscalibration.ipynb](CaseStudy_miscalibration.ipynb)


## Installation 
Download the repository and install the dependencies in a virtual environment using the [requirements.yml](requirements.yml) file.
Afterwards run `pip install netcal --no-deps` .Netcal includes outdates dependencies, however these are not used in the article.

If installed without `--no-deps` packages might be downgraded making the rest of the installation unusable.

## Citation
If used, please cite:

Jelke Wibbeke, Nico Schönfisch, Sebastian Rohjans, Andreas Rauh,
Evaluating the quality of the quantified uncertainty for (Re)calibration of data-driven regression models,
International Journal of Approximate Reasoning, Volume 195, 2026, 109685, ISSN 0888-613X, https://doi.org/10.1016/j.ijar.2026.109685.

```
@article{WIBBEKE2026109685,
title = {Evaluating the quality of the quantified uncertainty for (Re)calibration of data-driven regression models},
journal = {International Journal of Approximate Reasoning},
volume = {195},
pages = {109685},
year = {2026},
issn = {0888-613X},
doi = {https://doi.org/10.1016/j.ijar.2026.109685},
url = {https://www.sciencedirect.com/science/article/pii/S0888613X26000617},
author = {Jelke Wibbeke and Nico Schönfisch and Sebastian Rohjans and Andreas Rauh},
keywords = {Uncertainty quantification, Supervised learning, Ensembles, Likelihood, Review, Benchmark},
abstract = {In safety-critical applications data-driven models must not only be accurate but also provide reliable uncertainty estimates. This property, commonly referred to as calibration, is essential for risk-aware decision-making. In regression a wide variety of calibration metrics and recalibration methods have emerged. However, these metrics differ significantly in their definitions, assumptions and scales. As a result, interpreting and comparing results across studies becomes challenging. Moreover, most recalibration methods have been evaluated using only a small subset of metrics, leaving it unclear whether improvements generalize across different notions of calibration. In this work, we systematically extract and categorize regression calibration metrics from the literature. We benchmark these metrics independently of specific modelling methods or recalibration approaches. Through controlled experiments with real-world, synthetic and artificially miscalibrated data, we demonstrate that calibration metrics frequently produce conflicting results. Our analysis reveals substantial inconsistencies: many metrics disagree in their evaluation of the same recalibration result, and some even indicate contradictory conclusions. This inconsistency is particularly concerning as it allows cherry-picking of metrics to create misleading impressions of success. We identify that the Expected Normalized Calibration Error (ENCE) and the Coverage Width-based Criterion (CWC) are the most dependable metrics within our Gaussian uncertainty-based test framework. The results highlight the critical role of metric selection in calibration research.}
}
```
