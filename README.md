# ParabolicSolvationEnergies

# About
ParabolicSolvationEnergies allows to calculate donor number values using conceptual Density Functional Theory methods.
The basic theory is detailed in: "Quantitative Solvation Energies from Gas-Phase Calculations: First Principles Charge Transfer and Perturbation Approaches",
R. A. Miranda-Quintana, V. S. J. Craig, J. Smiatek; Chem. Comms. (submitted)

# License
ParabolicSolvationEnergies is distributed under GPL License version 3 (GPLv3).

# Dependencies
Python >= 3.3;  http://www.python.org/

Numpy >= 1.9.1;  http://www.numpy.org/

SciPy >= 0.11.0;  http://www.scipy.org/

Matplotlib >= 1.0;  http://matplotlib.org/

# Usage
The file solvation_fit.py provides the functionality to fit the experimental data using C-DFT methods.
The file solvation_cv.py allows to run the cross-validation tests.
The file	solvation_cv_output.py generates the output files.
The experimental values must be provided in a separate file containing three columns separated by ",":

solvent,cation,anion,dH-+,dHsol,dG-+,dGsol

Currently, the ionization energies and electron affinities are included as dictionaries in the solvation_cv_output.py file.

# Reference
Please, cite both the associated manuscript:

"Quantitative Solvation Energies from Gas-Phase Calculations: First Principles Charge Transfer and Perturbation Approaches",
R. A. Miranda-Quintana, V. S. J. Craig, J. Smiatek; Chem. Comms. (submitted)

And this repository:

DOI: (to be added after each release)

# Further reading
Some relevant references are:

1- C-DFT-based solvation models:

"Enthalpic contributions to solvent–solute and solvent–ion interactions: Electronic perturbation as key to the understanding of molecular attraction", J. Smiatek, J. Chem. Phys., 150, 174112, (2019).

"Specific Ion Effects and the Law of Matching Solvent Affinities: A Conceptual Density Functional Theory Approach", J. Smiatek, J. Phys. Chem. B, 124, 2191, (2020).

2- Perturbations in C-DFT:

"Fractional electron number, temperature, and perturbations in chemical reactions", R. A. Miranda-Quintana, P. W. Ayers, Phys. Chem. Chem. Phys., 18, 15070, (2016).

"Perturbed reactivity descriptors: the chemical hardness", R. A. Miranda-Quintana, Theo. Chem. Acc., 136, 76, (2017).
