# Evaluating the Quality of the Quantified Uncertainty for (Re)Calibration of Data-Driven Regression Models
This repository contains the code to reproduce the benchmark of our paper [Evaluating the Quality of the Quantified Uncertainty for (Re)Calibration of Data-Driven Regression Models]().


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

Wibbeke, J., Schönfisch, N., Rohjans, S. and Rauh, A. (2025), Evaluating the Quality of the Quantified Uncertainty for (Re)Calibration of Data-Driven Regression Models
```
@article{wibbeke2024quantification,
author = {Wibbeke, Jelke and Sch{\"o}nfisch, Nico and Rohjans, Sebastian and Rauh, Andreas},
title = {Evaluating the Quality of the Quantified Uncertainty for (Re)Calibration of Data-Driven Regression Models},
year = {2025},
}
```