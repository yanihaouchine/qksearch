# qksearch — automatic search for quantum kernel circuits

Quantum kernel methods live or die by their feature map, and that map is
usually designed by hand. This project treats the design as a **combinatorial
search**: it builds IQP circuits gate by gate and scores each candidate with
the **kernel-target alignment** (KTA) of its Gram matrix, optionally
regularized by the dimension of the circuit's dynamical Lie algebra or by the
spread of the Gram matrix. It is then applied to regression of gas adsorption
in metal-organic frameworks (MOF).

Research internship at the Prasad Research Group, University of Calgary (2026).

