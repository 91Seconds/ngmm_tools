# Tools for Developing Nonergodic Ground Motion Models

This repository contains software tools for developing Nonergodic Ground Motion Models (NGMMs) based on the varying coefficient (Landwehr et al., 2016) and cell-specific anelastic attention approach (Dawood and Rodriguez‐Marek, 2013). 
Developed tools are are available in R using the statistical package (R-INLA, https://www.r-inla.org/) and in python using (Stan, https://mc-stan.org/).

## Folder Structure
The main folder ``Analyses`` contains all the regression, testing, and library scripts. 
It is divided into ``Data_Preparation`` which includes preprocessing scripts for the NGMM regression, ``Regression`` which contains the Jupyter notebooks for running the NGMM regressions using Stan and INLA, ``Predictions`` which contains conditional predictions for new scenarios based on the regression results, and ``Code_Verification`` which contains the scripts associated with the verification exercise. 
Folders ``Python_lib``, ``R_lib``, and ``Stan_lib`` contain scripts used in the developed tools.

The main folder ``Data`` contains all the files generated by the tools in ``Analyses``.

    .
    |--Analyses
    |     |--Data_Preparation
    |     |--Regression
    |     |--Predictions
    |     |--Code_Verification
    |     |--Python_lib
    |     |--R_lib
    |     |--Stan_lib
    |
    |--Data
    |     |--Regression
    |     |--Predictions
    |     |--Code_Verification
    |     
    |--Raw_files

## Additional Information
Additional information on the development of NGMMs with the proposed approach can be found at Lavrentiadis et al. (In Press - Available in ArXiv) and NHR3 report (published by May, 2022).

## Acknowledgments 
Financial support by the California Department of Transportation and Pacific Gas & Electric Company is greatly appreciated.  

## References
Dawood, H. M., & Rodriguez‐Marek, A. (2013). A method for including path effects in ground‐motion prediction equations: An example using the M w 9.0 Tohoku earthquake aftershocks. Bulletin of the Seismological Society of America, 103(2B), 1360-1372.

Landwehr, N., Kuehn, N. M., Scheffer, T., & Abrahamson, N. (2016). A nonergodic ground‐motion model for California with spatially varying coefficients. Bulletin of the Seismological Society of America, 106(6), 2574-2583.

Lavrentiadis, G., Abrahamson, N. A., Nicolas, K. M., Bozorgnia, Y., Goulet, C. A., Babič, A., ... & Walling, M. (2021). Overview and Introduction to Development of Non-Ergodic Earthquake Ground-Motion Models. arXiv preprint arXiv:2111.07921.
