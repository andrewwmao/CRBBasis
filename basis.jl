using Pkg
Pkg.activate(".")
Pkg.instantiate()
using LinearAlgebra
using MAT
using Printf

println("Threads=$(Threads.nthreads())"); flush(stdout)
ijob = parse(Int32, ENV["SLURM_ARRAY_TASK_ID"]) # for a job submitted to a cluster managed by Slurm
λ = ijob / 10 # assumes ijob goes from 0-10
println("lambda=$(λ)"); flush(stdout);

data_path = "fingerprints_ograd/"
Nfiles = [16, 2, 2]

println(string("Basis: ", data_path)); flush(stdout)

## Checks if all files are present
iFile = [1:Nfiles[1]; 31:30+Nfiles[2]; 41:40+Nfiles[3]]
fileavailable = trues(sum(Nfiles))
for i in iFile
    data_file = string(data_path, "ijob", i, ".mat")
    if !isfile(data_file)
        fileavailable[i] = false
        println(string("ijob", i, " does not exist"))
    end
end
idx = iFile[findall(x->x==true,fileavailable)]

## Load Data
@info "Loading fingerprints"
flush(stderr)

s_temp = matread(string(data_path, "ijob1.mat"))["s"] # find dimensions
s = zeros(ComplexF32, size(s_temp, 1), size(s_temp,2), size(s_temp,3), sum(Int.(fileavailable)), 2)
@time for i in eachindex(idx)
    data_file = string(data_path, "ijob", idx[i], ".mat")
    file = matopen(data_file)
    s[:,:,:,i,1] .= read(file, "s")
    close(file)

    # multiply λ
    s[:,1,:,i,1] .*= (1 - λ)
    s[:,2:end,:,i,1] .*= λ

    # add conjugate
    s[:,:,:,i,2] .= conj.(s[:,:,:,i,1])
end

flush(stdout)
if λ == 0 #cutout the zero parts to speed things up
    s = s[:,1,:,:,:]
elseif λ == 1
    s = s[:,2:end,:,:,:]
end
s = reshape(s, size(s_temp,1), :) #reshape before svd

## Calculate CRB-SVD
BLAS.set_num_threads(Threads.nthreads())
@info "Starting SVD"
flush(stderr)
t = @elapsed U, S, _ = svd!(s)
flush(stdout)
GC.gc()

## Save Results
@info "Saving svd to file"
flush(stderr)

file = matopen(@sprintf("basis_λ%.2f.mat", λ), "w")
write(file, "U", real.(U[:,1:50]));
write(file, "S", S);
write(file, "t", t);
close(file);