#TODO: don't add ograd for fat/CSF
#TODO: conjugate random half of samples instead of doubling sample size

using Pkg
Pkg.activate(".")
Pkg.instantiate()
using LinearAlgebra
using MAT

println("Threads=$(Threads.nthreads())"); flush(stdout)
ijob = parse(Int32, ENV["SLURM_ARRAY_TASK_ID"])
lambda = min(1, max(0, mod(ijob, 10) / 10)) # assume ijob goes from 0-9
println("lambda=$(lambda)"); flush(stdout);

data_path = "fingerprints_ograd/"
save_path = "../"
Nfiles = [160, 20, 20]

println(string("Basis: ", data_path)); flush(stdout)

# check all files are present
iFile = [1:Nfiles[1]; 201:200+Nfiles[2]; 301:300+Nfiles[3]]
fileavailable = trues(sum(Nfiles))
for i in iFile
    data_file = string(data_path, "ijob", i, ".mat")
    if !isfile(data_file)
        fileavailable[i] = false
        println(string("ijob", i, " does not exist"))
    end
end
idx = iFile[findall(x->x==true,fileavailable)]

# load data
@info "Loading fingerprints"
flush(stderr)

s_temp = matread(string(data_path, "ijob1.mat"))["s"] # find dimensions
s = zeros(ComplexF32, size(s_temp, 1), size(s_temp,2), size(s_temp,3), sum(Int.(fileavailable)), 2)
@time for i in eachindex(idx)
    data_file = string(data_path, "ijob", idx[i], ".mat")
    file = matopen(data_file)
    s[:,:,:,i,1] .= read(file, "s") #using straight matread here seems to cause errors
    close(file)

    # multiply lambda
    s[:,1,:,i,1] .*= (1 - lambda)
    s[:,2:end,:,i,1] .*= lambda

    # add conjugate
    s[:,:,:,i,2] .= conj.(s[:,:,:,i,1])
end

flush(stdout)
if lambda == 0 #cutout the zero parts to speed things up
    s = s[:,1,:,:,:]
elseif lambda == 1
    s = s[:,2:end,:,:,:]
end
s = reshape(s, size(s_temp,1), :) #reshape before svd

# calculate CRB-SVD
BLAS.set_num_threads(Threads.nthreads())
@info "Starting SVD"
flush(stderr)
t = @elapsed U, S, _ = svd!(s)
flush(stdout)
GC.gc()

@info "Saving svd to file"
flush(stderr)

if (length(ARGS) > 0) && parse(Bool, ARGS[1]) # if isbrain
    file_str = "_brain"
else
    file_str = "_broad"
end

file = matopen("$(save_path)/basis_MT$(file_str).mat", "w")
write(file, "U", real.(U[:,1:100]));
write(file, "S", S);
write(file, "t", t);
close(file);

## calculate training dataset
@info "Calculating training dataset"
flush(stderr)

files = 1:400
Nfiles = length(files)
TR = 3.5e-3
R = 15
if (length(ARGS) > 0) && parse(Bool, ARGS[1]) # if isbrain
    file_str = "_brain"
else
    file_str = "_broad"
end
U = matread("../basis_MT$(file_str).mat")["U"][:,1:R]

data_file = "fingerprints/ijob$(files[1]).mat"
file = matopen(data_file)
s_temp = read(file, "s")
p_temp = read(file, "p")
close(file)
Nbatch = size(p_temp, 2)

# merge into one preprocessed td_file
sc = zeros(ComplexF32, R, Nbatch, Nfiles)
p = zeros(Float64, size(p_temp,1), Nbatch, Nfiles)
CRBc = zeros(Float64, size(p,1), Nbatch, Nfiles)
@time for i in eachindex(files) #threading seems slower
    data_file = string("fingerprints/ijob", files[i], ".mat")
    try
        file = matopen(data_file)
        s_temp = read(file, "s")
        p[:,:,i] .= read(file, "p")
        close(file)
        for j = 1:Nbatch
            s_temp2 = U' * @view(s_temp[:,:,j]) # calculate compressed CRB
            sc[:,j,i] .= s_temp2[:,1]
            D = zeros(size(s_temp2,2))
            FIM = s_temp2' * s_temp2
            for k = 1:size(p,1)
                D[1+k] = 1
                CRB = FIM \ D
                CRBc[k,j,i] = real.(CRB)[1+k]
                D[1+k] = 0
            end
        end
    catch
        println(string("ijob", i, " does not exist"))
    end
    GC.gc()
end
sc = reshape(sc, R, :)
p = reshape(p, size(p_temp,1), :)
CRBc = reshape(CRBc, size(p,1), :)
@views p[7,:] .= abs.(p[7,:]) ./ (π/TR) #normalize range of w0 to [-1,1]
@views CRBc[7,:] ./= (π/TR)^2

## correct for missing jobs
idx_nonzero = findall(!iszero, p[2,:])
if length(idx_nonzero) < size(sc,2)
    sc = sc[:, idx_nonzero]
    p = p[:, idx_nonzero]
    CRBc = CRBc[:, idx_nonzero]
    GC.gc()
end

@info "Saving training dataset"
flush(stderr)

file = matopen("td_MT_Nc$(R).mat", "w");
write(file, "sc", sc);
write(file, "p", p);
write(file, "CRB", CRBc);
close(file);

file = matopen("td_MT_Nc$(R)_small.mat", "w");
write(file, "sc",    sc[:,1:5:end]);
write(file, "p",      p[:,1:5:end]);
write(file, "CRB", CRBc[:,1:5:end]);
close(file);