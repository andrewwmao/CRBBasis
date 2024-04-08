# Cramer-Rao Bound Optimized Subspace Reconstruction in Quantitative MRI

This repository shows how to optimize the CRB-SVD bases described in https://arxiv.org/abs/2305.00326.

In sim.jl, a dictionary is computed using the [MRIgeneralizedBloch.jl](https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl) package. For each fingerprint, the signal derivatives are then orthogonalized with respect to one another using QR factorization.

In basis.jl, the signals and orthogonalized derivatives are loaded into a large matrix, on which the SVD is performed.

Both scripts are provided with example bash scripts that can be used to submit the jobs to a computational cluster, in this case managed by Slurm.

Our preprint is currently under revision at IEEE Transactions on Biomedical Engineering. This repository will be finalized when the paper is published.