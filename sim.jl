using Pkg
Pkg.activate(".")
Pkg.instantiate()
using LinearAlgebra
using MAT
using Random
using MRIgeneralizedBloch
using ProgressBars

## parameters
Nfp = 2^11 #2048
TR = 3.5e-3

T2smin = 5e-6
T2smax = 25e-6
B1min = 0.6
B1max = 1.3
TRF_max = 500e-6
idx_grad = [2,3,4,5,6,7] # which gradients should be orthogonalized

## load control, basis files
control = matread("3T_v10p9_pattern.mat")
α = [reshape(control["alpha"],:,6)[:,i] for i = 1:size(reshape(control["alpha"],:,6),2)]
TRF = [reshape(control["TRF"],:,6)[:,i] for i = 1:size(reshape(control["TRF"],:,6),2)]

R2slT = precompute_R2sl(T2s_min=T2smin, T2s_max=T2smax, B1_max=B1max,TRF_max=TRF_max)
grad_list = [grad_m0s(), grad_R1f(), grad_R2f(), grad_Rx(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1()]

ijob = try
    parse(Int32, ENV["SLURM_ARRAY_TASK_ID"])
catch
    1
end
rng = MersenneTwister(ijob);
println("ijob = $ijob")

##
if (length(ARGS) > 0) && parse(Bool, ARGS[1]) # if isbrain
    function set_rand_parameters(rng, T2smin, T2smax, B1min, B1max) # brain parenchyma
        m0s = Inf
        while (m0s < 0 || m0s > 0.35)
            m0s = 0.15 + 0.15 * randn(rng, Float64)
        end

        R1f = Inf
        while (R1f < 0.1 || R1f > 1)
            R1f = 0.45 + 0.2 * randn(rng, Float64)
        end

        R2f = Inf
        while (R2f < 1 || R2f > 20 || R2f < R1f)
            R2f = 12.5 + 4 * randn(rng, Float64)
        end

        Rx = Inf
        while (Rx < 2 || Rx > 30)
            Rx = 10 + 5 * randn(rng, Float64)
        end

        R1s = Inf
        while (R1s < 1 || R1s > 6)
            R1s = 3 + 0.75 * randn(rng, Float64)
        end

        T2s = Inf
        while (T2s < T2smin || T2s > T2smax)
            T2s = 14e-6 + 4e-6 * randn(rng, Float64)
        end

        ω0 = π/TR * (-1 + 2 * rand(rng, Float64))
        B1 = Inf
        while (B1<B1min || B1>B1max)
            B1 = 0.9 + 0.2 * randn(rng, Float64)
        end
        return (m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1)
    end

    function set_rand_params_fat(rng, B1min, B1max)
        m0s = 1e-3
        Rx = 10
        R1s = 3
        T2s = 12e-6

        T1f = Inf
        while (T1f < 0.1 || T1f > 0.7)
            T1f = 0.4 + 0.075 * randn(rng, Float64) #fast relaxation
        end
        R1f = 1/T1f

        T2f = Inf
        while (T2f < 0.05 || T2f > 0.2)
            T2f = 0.1 + 0.02 * randn(rng, Float64) #fast relaxation
        end
        R2f = 1/T2f

        ω0 = π/TR * (-1 + 2 * rand(rng, Float64))
        B1 = Inf
        while (B1<B1min || B1>B1max)
            B1 = 0.75 + 0.125 * randn(rng, Float64) #lower B1 expected for skull fat
        end
        return (m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1)
    end

    function set_rand_params_CSF(rng, B1min, B1max)
        m0s = 1e-3
        Rx = 10
        R1s = 3
        T2s = 12e-6

        T1f = Inf
        while (T1f < 1 || T1f > 7)
            T1f = 4 + 0.75 * randn(rng, Float64)
        end
        R1f = 1/T1f

        T2f = Inf
        while (T2f < 0.5 || T2f > 5)
            T2f = 2 + 1 * randn(rng, Float64)
        end
        R2f = 1/T2f

        ω0 = π/TR * (-1 + 2 * rand(rng, Float64))
        B1 = Inf
        while (B1<B1min || B1>B1max)
            B1 = 0.9 + 0.3 * randn(rng, Float64)
        end
        return (m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1)
    end

else

    function set_rand_parameters(rng, T2smin, T2smax, B1min, B1max)
        m0s = Inf
        while (m0s < 0 || m0s > 0.35)
            m0s = 0.15 + 0.15 * randn(rng, Float64)
        end

        R1f = Inf
        while (R1f < 0.1 || R1f > 2)
            R1f = 0.75 + 0.45 * randn(rng, Float64)
        end

        R2f = Inf
        while (R2f < 3 || R2f > 45 || R2f < R1f)
            R2f = 24 + 10.5 * randn(rng, Float64)
        end

        Rx = Inf
        while (Rx < 5 || Rx > 50)
            Rx = 30 + 10 * randn(rng, Float64)
        end

        R1s = Inf
        while (R1s < 1 || R1s > 6)
            R1s = 3 + 2randn(rng, Float64)
        end

        T2s = Inf
        while (T2s < T2smin || T2s > T2smax)
            T2s = 12e-6 + 5e-6 * randn(rng, Float64)
        end

        ω0 = π/TR * (-1 + 2 * rand(rng, Float64))
        B1 = Inf
        while (B1<B1min || B1>B1max)
            B1 = 0.9 + 0.3 * randn(rng, Float64)
        end
        return (m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1)
    end

    function set_rand_params_fat(rng, B1min, B1max)
        m0s = 1e-3
        Rx = 10
        R1s = 3
        T2s = 12e-6

        T1f = Inf
        while (T1f < 0.1 || T1f > 0.7)
            T1f = 0.4 + 0.075 * randn(rng, Float64) #fast relaxation
        end
        R1f = 1/T1f

        T2f = Inf
        while (T2f < 0.05 || T2f > 0.2)
            T2f = 0.1 + 0.02 * randn(rng, Float64) #fast relaxation
        end
        R2f = 1/T2f

        ω0 = π/TR * (-1 + 2 * rand(rng, Float64))
        B1 = Inf
        while (B1<B1min || B1>B1max)
            B1 = 0.75 + 0.125 * randn(rng, Float64) #lower B1 expected for skull fat
        end
        return (m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1)
    end

    function set_rand_params_CSF(rng, B1min, B1max)
        m0s = 1e-3
        Rx = 10
        R1s = 3
        T2s = 12e-6

        T1f = Inf
        while (T1f < 1 || T1f > 7)
            T1f = 4 + 0.75 * randn(rng, Float64)
        end
        R1f = 1/T1f

        T2f = Inf
        while (T2f < 0.5 || T2f > 5)
            T2f = 2 + 1 * randn(rng, Float64)
        end
        R2f = 1/T2f

        ω0 = π/TR * (-1 + 2 * rand(rng, Float64))
        B1 = Inf
        while (B1<B1min || B1>B1max)
            B1 = 0.9 + 0.3 * randn(rng, Float64)
        end
        return (m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1)
    end
end

function calc_training_data(Nfp, α, TRF, TR, grad_list, R2slT, rng, T2smin, T2smax, B1min, B1max, ijob)
    s = zeros(ComplexF32, length(α[1]), length(α), length(grad_list)+1, Nfp)
    p = zeros(length(grad_list), Nfp) #8 parameters in split R1 model excluding M0
    iter = ProgressBar(1:Nfp)
    flush(stderr)
    for i in iter
        if ijob > 300
            (m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1) = set_rand_params_fat(rng, B1min, B1max)
        elseif ijob > 200
            (m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1) = set_rand_params_CSF(rng, B1min, B1max)
        else
            (m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1) = set_rand_parameters(rng, T2smin, T2smax, B1min, B1max)
        end
        p[:,i] = [m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1]
        #simulate using every control and concatenate
        for j ∈ axes(α,1)
            si = calculatesignal_linearapprox(α[j], TRF[j], TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT; grad_list=grad_list)
            s[:,j,:,i] .= reshape(si, (length(α[j]), length(grad_list)+1))
        end
        flush(stderr)
    end
    s = reshape(s, :, length(grad_list)+1, Nfp)
    return (s,p)
end

## training data
@info "Simulating fingerprints"
flush(stderr)
@time s, p = calc_training_data(Nfp, α, TRF, TR, grad_list, R2slT, rng, T2smin, T2smax, B1min, B1max, ijob)

file = matopen("fingerprints/ijob$(ijob).mat", "w")
write(file, "s", s)
write(file, "p", p)
close(file)

## compute ograd
@info "Calculating ograds"
flush(stderr)

si = zeros(ComplexF32, size(s, 1), length(idx_grad) + 1, size(s,3))
s_temp = zeros(ComplexF32, size(si, 1))
si[:,1,:] .= s[:,1,:] #copy signal
@time for i ∈ axes(s,3) #for batchsize
    for j ∈ eachindex(idx_grad) #for each interested gradient
        # swap interested gradient to last row
        s_temp .= s[:,end,i] #copy the last row
        s[:,end,i] .= s[:,idx_grad[j],i] #write the interested gradient into last row
        s[:,idx_grad[j],i] .= s_temp #shift last row into new spot

        #QR factorization
        Q,R = qr(@view(s[:,:,i])) #QR factorization on this sample
        si[:,j+1,i] .= Q[:,size(R,2)] #save the unit length orthogonalized gradient

        # undo the swap
        s_temp .= s[:,end,i] #copy the interested gradient
        s[:,end,i] .= s[:,idx_grad[j],i] #write last gradient back into last row
        s[:,idx_grad[j],i] .= s_temp #write interested gradient back
    end
end

file = matopen("fingerprints_ograd/ijob$(ijob).mat", "w")
write(file, "s", si)
close(file)

exit()