# Cramér-Rao Bound Optimized Subspace Reconstruction in Quantitative MRI

This repository shows how to optimize CRB-SVD bases as described in the paper https://arxiv.org/abs/2305.00326. These bases are useful for subspace reconstruction tasks performed as an intermediate step before parameter estimation in quantitative MRI.

In ```sim.jl```, a dictionary is computed using the [MRIgeneralizedBloch.jl](https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl) package. For each fingerprint, the signal derivatives are then orthogonalized with respect to one another using QR factorization.

In ```basis.jl```, the signals and orthogonalized derivatives are loaded into a large matrix on which the SVD is performed.

Example bash scripts are provided that can be used to submit both scripts as jobs to a computational cluster, in this case one managed by Slurm.

Our preprint is currently under revision at IEEE Transactions on Biomedical Engineering. This repository will be finalized when the paper is published.