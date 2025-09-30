module EDModule

using LinearAlgebra
using SparseArrays
using Random
using KrylovKit
using LinearMaps
using Distributed
using Combinatorics

export run_ed_simulation, ensure_worker_basis!

# =============================================================================
#                               Helpers & Types
# =============================================================================

"""
    _UintFor(L::Int) -> Type{UInt64 or UInt128}

Choose the minimal unsigned integer type that can hold L occupation bits.
- Uses UInt64 up to 64 modes, otherwise UInt128.
"""
@inline _UintFor(L::Int) = L <= 64 ? UInt64 : UInt128

"""
    mode_index(site::Int, flavor::Int, N::Int) -> Int

Map a (site, flavor) pair to a single 1-based mode index in row-major order.

Arguments:
- site   ∈ 1..M
- flavor ∈ 1..N
- N      number of flavors per site
"""
@inline function mode_index(site::Int, flavor::Int, N::Int)
    return (site - 1) * N + flavor
end

"""
    state_to_code(occ::Vector{Bool}) -> UInt64/UInt128

Convert a boolean occupation vector (LSB corresponds to index 1)
to an integer bit code. The integer width is chosen by _UintFor.
"""
@inline function state_to_code(occ::Vector{Bool})
    T = _UintFor(length(occ))
    code = zero(T)
    @inbounds for i in 1:length(occ)
        if occ[i]
            code |= (one(T) << (i - 1))
        end
    end
    return code
end

"""
    compute_charge_label(occ, M, N) -> Int

Compute a base-3 (QB convention) sector label
    Q(occ) = Σ_{m=1..M} 3^(M-m) * n_m,
where n_m is the number of occupied flavors at site m.

Notes:
- Keeps the original base-3 convention for compatibility (even if N ≠ 3).
- If you later need collision-free labels for arbitrary N, consider base-(N+1).
"""
function compute_charge_label(occ::Vector{Bool}, M::Int, N::Int)
    q = 0
    base_idx = 0
    @inbounds for m in 1:M
        n_m = 0
        @inbounds for _ in 1:N
            base_idx += 1
            n_m += occ[base_idx] ? 1 : 0
        end
        q += 3^(M - m) * n_m
    end
    return q
end

"""
    levi_cevita_sign(indices::NTuple{K,Int}) where K -> Int

General Levi-Civita sign for an index tuple.
- Returns 0 if any indices repeat.
- Returns (-1)^{inversions} otherwise.
"""
function levi_cevita_sign(indices::NTuple{K,Int}) where K
    v = collect(indices)
    n = length(v)
    if length(Set(v)) != n
        return 0
    end
    invs = 0
    @inbounds for i in 1:n
        @inbounds for j in (i+1):n
            invs += (v[i] > v[j]) ? 1 : 0
        end
    end
    return (-1)^invs
end

# -----------------------------------------------------------------------------
# Exact fermionic sign helpers operating on bitstring codes and occupation vecs
# -----------------------------------------------------------------------------

"""
    apply_annihilation(code, occ, p) -> (newcode, sign, ok)

Apply c_p on a Fock basis state represented by (code, occ).

Returns:
- newcode (integer)
- sign    (+1 or -1) from fermionic reordering
- ok      false if the mode was empty (annihilation failed)
"""
@inline function apply_annihilation(code::T, occ::Vector{Bool}, p::Int) where {T<:Unsigned}
    if !occ[p]; return (code, 1, false); end
    # Count occupied modes before p to get the fermionic sign
    cnt = 0
    @inbounds for q in 1:p-1
        cnt += occ[q] ? 1 : 0
    end
    newcode = code & ~(one(T) << (p - 1))
    occ[p] = false
    return (newcode, (-1)^cnt, true)
end

"""
    apply_creation(code, occ, p) -> (newcode, sign, ok)

Apply c_p^† on a Fock basis state represented by (code, occ).

Returns:
- newcode (integer)
- sign    (+1 or -1) from fermionic reordering
- ok      false if the mode was already occupied (creation failed)
"""
@inline function apply_creation(code::T, occ::Vector{Bool}, p::Int) where {T<:Unsigned}
    if occ[p]; return (code, 1, false); end
    cnt = 0
    @inbounds for q in 1:p-1
        cnt += occ[q] ? 1 : 0
    end
    newcode = code | (one(T) << (p - 1))
    occ[p] = true
    return (newcode, (-1)^cnt, true)
end

# -----------------------------------------------------------------------------
# J tensor construction (QB model) with optional randomness
# -----------------------------------------------------------------------------

"""
    generate_Jijkl(N, J0; model="QB", random_J=false, rng=Random.default_rng()) -> Array{Float64,4}

Construct the QB three-fermion coupling tensor J_{i j k l} (size N×N×N×N).

- If random_J == false (default, deterministic):
    J_{i j k l} = sgn(perm(i,j,k)) * J0  for all l.

- If random_J == true:
    For each unordered triplet (i<j<k) and each l:
      draw A ~ 𝒩(0, variance = J0)  (i.e., std = √J0),
      then set J_{π(i) π(j) π(k) l} = sgn(π) * A for all permutations π.
    This preserves total antisymmetry in (i,j,k) and gives l-resolved randomness.

Arguments:
- N   : number of flavors per site
- J0  : coupling scale (or variance if random_J=true)
- rng : RNG to control randomness when random_J=true
"""
function generate_Jijkl(N::Int, J0::Number; model::String="QB", random_J::Bool=false, rng=Random.default_rng())
    J = zeros(Float64, N, N, N, N)
    if model != "QB"
        @warn "Only 'QB' model is implemented; returning zero tensor for others."
        return J
    end
    if N < 3
        @warn "N must be at least 3 for Jijkl to be non-zero for 'QB' model. J will be zero."
        return J
    end

    if !random_J
        # Deterministic antisymmetric tensor
        for (i0, j0, k0) in permutations(0:N-1, 3)
            s = levi_cevita_sign((i0, j0, k0))
            @inbounds for l0 in 0:N-1
                J[i0+1, j0+1, k0+1, l0+1] = s * Float64(J0)
            end
        end
    else
        # Random antisymmetric tensor: variance = J0 (std = sqrt(J0))
        σ = sqrt(Float64(J0))
        # For each unordered (i<j<k) and each l, draw base A and populate permutations
        for i in 1:N-2, j in i+1:N-1, k in j+1:N, l in 1:N
            A = σ * randn(rng)
            for (a,b,c) in permutations((i,j,k))
                s = levi_cevita_sign((a-1, b-1, c-1))
                J[a, b, c, l] = s * A
            end
        end
    end
    return J
end

# -----------------------------------------------------------------------------
# Cached basis structs
# -----------------------------------------------------------------------------

"""
    SectorBasis

Holds the explicit Fock basis for a fixed charge-Q sector.
Fields:
- states   :: Vector of Bool occupation vectors
- codes    :: Matching integer bit-codes for states
- index_of :: Dict mapping code -> basis index
"""
struct SectorBasis{T<:Unsigned}
    states   :: Vector{Vector{Bool}}
    codes    :: Vector{T}
    index_of :: Dict{T,Int}
end

"""
    BasisByQ

Cache of the full basis grouped by the QB charge label Q.
Fields:
- M, N, L : geometry (sites, flavors, total modes)
- full_Q0     : normalization factor used for reporting Q
- sectors     : Dict{Q, SectorBasis}
"""
struct BasisByQ{T<:Unsigned}
    M        :: Int
    N        :: Int
    L        :: Int
    full_Q0  :: Int
    sectors  :: Dict{Number, SectorBasis{T}}
end

# Per-process cache: (M, N) -> BasisByQ
const WORKER_BASIS_CACHE = Dict{Tuple{Int,Int}, Any}()

"""
    _build_basis_by_Q(M, N) -> BasisByQ

Enumerate all 2^(M*N) Fock states, bucket by the QB-convention charge label Q,
and build per-sector bases with (state, code, index) mappings.

Warning:
- This brute-force enumeration is only viable for small L = M*N.
"""
function _build_basis_by_Q(M::Int, N::Int)
    L = M * N
    Tcode = _UintFor(L)

    # Temporary buckets
    sector_states = Dict{Number, Vector{Vector{Bool}}}()
    sector_codes  = Dict{Number, Vector{Tcode}}()

    # Enumerate all states from bit patterns of UInt
    limit = (UInt(1) << L) - 1
    @inbounds for x in zero(UInt):limit
        # Build occ vector from bits of x
        occ = Vector{Bool}(undef, L)
        xx = x
        @inbounds for i in 1:L
            occ[i] = (xx & 0x01) == 0x01
            xx >>= 1
        end

        q  = compute_charge_label(occ, M, N)
        cd = state_to_code(occ)

        push!(get!(sector_states, q, Vector{Vector{Bool}}()), occ)
        push!(get!(sector_codes,  q, Tcode[]), cd)
    end

    # Finalize per-sector structures
    sectors = Dict{Number, SectorBasis{Tcode}}()
    for (q, occs) in sector_states
        cds = sector_codes[q]
        idx = Dict{Tcode,Int}(cds[i] => i for i in eachindex(cds))
        sectors[q] = SectorBasis{Tcode}(occs, cds, idx)
    end

    # Full charge for normalization in reports
    full_Q0 = N * sum(3^(M - m) for m in 1:M)
    return BasisByQ{Tcode}(M, N, L, full_Q0, sectors)
end

"""
    ensure_worker_basis!(M, N)

Ensure (build if needed) that the current process holds a cached BasisByQ
for (M, N). Safe to call multiple times; builds at most once per process.
"""
function ensure_worker_basis!(M::Int, N::Int)
    key = (M, N)
    if !haskey(WORKER_BASIS_CACHE, key)
        WORKER_BASIS_CACHE[key] = _build_basis_by_Q(M, N)
    end
    return nothing
end

@inline _basis(M::Int, N::Int)::BasisByQ = WORKER_BASIS_CACHE[(M, N)]::BasisByQ
@inline _sector(M::Int, N::Int, q)::SectorBasis = _basis(M, N).sectors[q]

# =============================================================================
#                       Sector Hamiltonian Construction
# =============================================================================

"""
    apply_sector_diagonal!(y, v, sb, M, N, μ, ν)

Accumulate the diagonal on-site contributions into y:
    H_diag = Σ_{m,i} (μ + ν_{g_{m,i}}) n_{g_{m,i}},
where g_{m,i} = mode_index(m, i, N).

- Operates in the Fock basis of sb.
- Adds diag .* v into y (i.e., y ← y + D*v).
"""
@inline function apply_sector_diagonal!(
    y::StridedVector{T}, v::StridedVector{T},
    sb::SectorBasis, M::Int, N::Int,
    μ::Float64, ν::Vector{Float64}
) where {T<:Real}
    @inbounds for row in 1:length(sb.states)
        occ = sb.states[row]
        diag = 0.0
        base = 0
        @inbounds for _m in 1:M
            @inbounds for _i in 1:N
                base += 1
                if occ[base]
                    diag += μ + ν[base]
                end
            end
        end
        y[row] += diag * v[row]
    end
    return nothing
end

"""
    apply_sector_offdiag_qb!(y, v, sb, M, N, J)

Accumulate the QB three-fermion off-diagonal processes into y:

1 → 3 (forward):
    J_{i j k l} c^†_{g_{m+1,i}} c^†_{g_{m+1,j}} c^†_{g_{m+1,k}} c_{g_{m,l}}

3 → 1 (Hermitian):
    J_{l k j i} c^†_{g_{m,l}} c_{g_{m+1,k}} c_{g_{m+1,j}} c_{g_{m+1,i}}

- Applies operator strings with exact fermionic signs by acting on occ and code.
- Uses i<j<k loops and builds all needed permutations via explicit creation order.
"""
function apply_sector_offdiag_qb!(
    y::StridedVector{T}, v::StridedVector{T},
    sb::SectorBasis, M::Int, N::Int,
    J::Array{Float64,4}
) where {T<:Real}

    index_of = sb.index_of
    states   = sb.states
    codes    = sb.codes

    @inbounds for row in 1:length(states)
        occ_from  = states[row]
        code_from = codes[row]
        amp       = v[row]

        @inbounds for m in 1:(M - 1)
            @inbounds for ℓ in 1:N
                idx_ℓ = mode_index(m, ℓ, N)

                # Iterate over ordered triplets i<j<k (per site m+1)
                @inbounds for i in 1:N, j in i+1:N, k in j+1:N
                    idx_i = mode_index(m + 1, i, N)
                    idx_j = mode_index(m + 1, j, N)
                    idx_k = mode_index(m + 1, k, N)

                    # ----- 1 → 3 term -----
                    if occ_from[idx_ℓ] && !(occ_from[idx_i] || occ_from[idx_j] || occ_from[idx_k])
                        code = code_from
                        occ  = copy(occ_from)

                        code1, s1, ok1 = apply_annihilation(code, occ, idx_ℓ);  ok1 || continue
                        code2, s2, ok2 = apply_creation(   code1, occ, idx_k);  ok2 || continue
                        code3, s3, ok3 = apply_creation(   code2, occ, idx_j);  ok3 || continue
                        code4, s4, ok4 = apply_creation(   code3, occ, idx_i);  ok4 || continue

                        to_idx = get(index_of, code4, 0)
                        if to_idx != 0
                            coupling = J[i, j, k, ℓ]
                            y[to_idx] += (coupling * s1 * s2 * s3 * s4) * amp
                        end
                    end

                    # ----- 3 → 1 term (Hermitian) -----
                    if !occ_from[idx_ℓ] && (occ_from[idx_i] && occ_from[idx_j] && occ_from[idx_k])
                        code = code_from
                        occ  = copy(occ_from)

                        code1, s1, ok1 = apply_annihilation(code,  occ, idx_i); ok1 || continue
                        code2, s2, ok2 = apply_annihilation(code1, occ, idx_j); ok2 || continue
                        code3, s3, ok3 = apply_annihilation(code2, occ, idx_k); ok3 || continue
                        code4, s4, ok4 = apply_creation(   code3, occ, idx_ℓ); ok4 || continue

                        to_idx = get(index_of, code4, 0)
                        if to_idx != 0
                            coupling = J[ℓ, k, j, i]  # J is real; hermitian conj. reflected by indices
                            y[to_idx] += (coupling * s1 * s2 * s3 * s4) * amp
                        end
                    end
                end
            end
        end
    end
    return nothing
end

"""
    sector_matvec!(y, x, sb, M, N, J, μ, ν)

Apply the sector Hamiltonian H to x (y ← H*x) by calling the
diagonal and off-diagonal accumulators.
"""
function sector_matvec!(
    y::StridedVector{Float64}, x::StridedVector{Float64},
    sb::SectorBasis, M::Int, N::Int,
    J::Array{Float64,4}, μ::Float64, ν::Vector{Float64}
)
    fill!(y, 0.0)
    apply_sector_diagonal!(y, x, sb, M, N, μ, ν)
    apply_sector_offdiag_qb!(y, x, sb, M, N, J)
    return y
end

# -----------------------------------------------------------------------------
# Dense / Krylov diagonalization per sector
# -----------------------------------------------------------------------------

"""
    dense_sector_matrix(sb, M, N, J, μ, ν) -> Hermitian{Float64, Matrix{Float64}}

Build the explicit dense Hermitian matrix for a given sector basis sb.
- This is only feasible for modest sector dimensions.
"""
function dense_sector_matrix(
    sb::SectorBasis, M::Int, N::Int,
    J::Array{Float64,4}, μ::Float64, ν::Vector{Float64}
)
    dim = length(sb.states)
    Is = Int[]; Js = Int[]; Vs = ComplexF64[]

    # Diagonal
    @inbounds for row in 1:dim
        occ = sb.states[row]
        diag = 0.0
        base = 0
        @inbounds for _m in 1:M
            @inbounds for _i in 1:N
                base += 1
                if occ[base]; diag += μ + ν[base]; end
            end
        end
        push!(Is, row); push!(Js, row); push!(Vs, diag)
    end

    # Off-diagonal
    index_of = sb.index_of
    states   = sb.states
    codes    = sb.codes

    @inbounds for row in 1:dim
        occ_from  = states[row]
        code_from = codes[row]

        @inbounds for m in 1:(M - 1)
            @inbounds for ℓ in 1:N
                idx_ℓ = mode_index(m, ℓ, N)
                @inbounds for i in 1:N, j in i+1:N, k in j+1:N
                    idx_i = mode_index(m + 1, i, N)
                    idx_j = mode_index(m + 1, j, N)
                    idx_k = mode_index(m + 1, k, N)

                    # 1 → 3
                    if occ_from[idx_ℓ] && !(occ_from[idx_i] || occ_from[idx_j] || occ_from[idx_k])
                        code = code_from; occ = copy(occ_from)
                        code1, s1, ok1 = apply_annihilation(code,  occ, idx_ℓ); ok1 || continue
                        code2, s2, ok2 = apply_creation(   code1, occ, idx_k); ok2 || continue
                        code3, s3, ok3 = apply_creation(   code2, occ, idx_j); ok3 || continue
                        code4, s4, ok4 = apply_creation(   code3, occ, idx_i); ok4 || continue

                        to_idx = get(index_of, code4, 0)
                        if to_idx != 0
                            coupling = J[i, j, k, ℓ]
                            push!(Is, row); push!(Js, to_idx); push!(Vs, coupling * s1 * s2 * s3 * s4)
                        end
                    end

                    # 3 → 1
                    if !occ_from[idx_ℓ] && (occ_from[idx_i] && occ_from[idx_j] && occ_from[idx_k])
                        code = code_from; occ = copy(occ_from)
                        code1, s1, ok1 = apply_annihilation(code,  occ, idx_i); ok1 || continue
                        code2, s2, ok2 = apply_annihilation(code1, occ, idx_j); ok2 || continue
                        code3, s3, ok3 = apply_annihilation(code2, occ, idx_k); ok3 || continue
                        code4, s4, ok4 = apply_creation(   code3, occ, idx_ℓ); ok4 || continue

                        to_idx = get(index_of, code4, 0)
                        if to_idx != 0
                            coupling = J[ℓ, k, j, i]
                            push!(Is, row); push!(Js, to_idx); push!(Vs, coupling * s1 * s2 * s3 * s4)
                        end
                    end
                end
            end
        end
    end

    Hs = sparse(Is, Js, Vs, dim, dim)
    return Hermitian(Matrix(Hs))
end

"""
    dense_sector_eigs(sb, M, N, J, μ, ν) -> Vector{Float64}

Return the full (sorted) spectrum of the sector Hamiltonian by building
its explicit dense matrix. (Feasible when sector dim is modest.)
"""
function dense_sector_eigs(
    sb::SectorBasis, M::Int, N::Int,
    J::Array{Float64,4}, μ::Float64, ν::Vector{Float64}
)::Vector{Float64}
    H = dense_sector_matrix(sb, M, N, J, μ, ν)
    evals = eigvals!(H)
    return real.(sort(evals))
end

"""
    dense_sector_groundstate(sb, M, N, J, μ, ν) -> (E0, ψ0)

Compute the ground-state eigenpair (value & vector) by dense diagonalization.
"""
function dense_sector_groundstate(
    sb::SectorBasis, M::Int, N::Int,
    J::Array{Float64,4}, μ::Float64, ν::Vector{Float64}
)::Tuple{Float64, Vector{Float64}}
    H = dense_sector_matrix(sb, M, N, J, μ, ν)
    F = eigen!(H)                  # Symmetric real; returns real eigenpairs
    idx = argmin(F.values)
    return (F.values[idx], F.vectors[:, idx])
end

"""
    krylov_sector_groundstate(sb, M, N, J, μ, ν; tol=1e-9, maxiter=50_000) -> (E0, ψ0)

Compute the ground-state eigenpair with a Krylov solver without forming H explicitly.
- Good for large sectors; uses sector_matvec!.
"""
function krylov_sector_groundstate(
    sb::SectorBasis, M::Int, N::Int,
    J::Array{Float64,4}, μ::Float64, ν::Vector{Float64};
    tol::Float64=1e-9, maxiter::Int=50_000
)::Tuple{Float64, Vector{Float64}}
    dim = length(sb.states)
    Hmap = LinearMap(Float64, dim, dim) do y, x
        sector_matvec!(y, x, sb, M, N, J, μ, ν)
    end
    vals, vecs, _ = KrylovKit.eigsolve(Hmap, randn(dim), 1; tol=tol, maxiter=maxiter, issymmetric=true)
    v = vecs[1]
    nv = norm(v); nv > 0 && (v = v ./ nv)
    return (float(vals[1]), collect(v))
end

"""
    sector_lowest_k_eigs(sb, M, N, J, μ, ν; k=8, dense_cutoff=3000, tol=1e-9, maxiter=50_000, is_target_Q=false)
        -> Vector{Float64}

Return the lowest k eigenvalues for the sector (sorted).
- If is_target_Q=true, return the **full spectrum** (dense).
- If dim ≤ dense_cutoff, compute dense; else use Krylov for k extremal vals.
"""
function sector_lowest_k_eigs(
    sb::SectorBasis, M::Int, N::Int,
    J::Array{Float64,4}, μ::Float64, ν::Vector{Float64};
    k::Int=8, dense_cutoff::Int=3000, tol::Float64=1e-9, maxiter::Int=50_000,
    is_target_Q::Bool=false
)
    dim = length(sb.states)
    if dim == 0
        return Float64[]
    end

    if is_target_Q
        @info "Performing dense diagonalization for full spectrum of target Q sector (Dim: $dim)"
        return dense_sector_eigs(sb, M, N, J, μ, ν)
    elseif dim <= dense_cutoff
        evals = dense_sector_eigs(sb, M, N, J, μ, ν)
        return length(evals) > k ? evals[1:k] : evals
    else
        Hmap = LinearMap(Float64, dim, dim) do y, x
            sector_matvec!(y, x, sb, M, N, J, μ, ν)
        end
        vals, _, _ = KrylovKit.eigsolve(Hmap, randn(dim), k;
                                        tol=tol, maxiter=maxiter, issymmetric=true)
        return sort!(collect(float.(vals)))
    end
end

# -----------------------------------------------------------------------------
# Densities
# -----------------------------------------------------------------------------

"""
    densities_from_state(sb, ψ, M, N) -> (per_site, per_mode)

Compute one-body densities from a normalized state vector ψ in basis sb.

Returns:
- per_site :: Vector{Float64} of length M, ⟨n_m⟩ summed over flavors at each site.
- per_mode :: Vector{Float64} of length M*N, ⟨n_g⟩ for each mode (flattened by site, then flavor).
"""
function densities_from_state(
    sb::SectorBasis, ψ::AbstractVector{<:Real}, M::Int, N::Int
)::Tuple{Vector{Float64}, Vector{Float64}}
    dim = length(sb.states)
    L = M * N
    @assert length(ψ) == dim "State length does not match sector dimension."

    # Accumulate ⟨n_g⟩ = Σ_α |ψ_α|^2 * n_g(α)
    per_mode = zeros(Float64, L)
    @inbounds for α in 1:dim
        w = ψ[α]^2
        if w == 0.0; continue; end
        occ = sb.states[α]
        @inbounds for g in 1:L
            if occ[g]; per_mode[g] += w; end
        end
    end

    # Per-site sums ⟨n_m⟩
    per_site = zeros(Float64, M)
    base = 0
    @inbounds for m in 1:M
        s = 0.0
        @inbounds for _i in 1:N
            base += 1
            s += per_mode[base]
        end
        per_site[m] = s
    end
    return (per_site, per_mode)
end

# =============================================================================
#                               Run Simulation
# =============================================================================

"""
    run_ed_simulation(M_val, N, J0, mu, W;
                      krylov_k=8, dense_cutoff=3000, krylov_tol=1e-9, krylov_maxiter=50_000,
                      target_Q_integer=-1, random_J=false, rng=Random.default_rng())

Run an exact-diagonalization sweep over QB charge-Q sectors.

Model:
- Three-fermion conversion between adjacent sites with on-site term:
    H = H_diag + H_QB
  where H_diag = Σ_{m,i} (μ + ν_g) n_g, and H_QB implements 1↔3 processes.

Couplings:
- If random_J=false: J_{i j k l} = sgn(perm(i,j,k)) * J0  (deterministic).
- If random_J=true : J is Gaussian within each l (charge flavor),
  antisymmetric in (i,j,k), mean 0, variance = J0.

Disorder:
- If W>0: ν_g ~ 𝒩(0, W^2) (seeded with 123 for reproducibility), else ν_g ≡ 0.

Returns a Dict with:
- "GS_energy"          : ground-state energy of the global minimum sector
- "GS_charge"          : that sector’s Q normalized by full_Q0
- "GS_densities"       : per-site densities ⟨n_m⟩ (length M_val)
- "GS_mode_densities"  : per-mode densities ⟨n_g⟩ (length M_val*N)
- "H_matrix_dim"       : total modes (M_val*N)
- "all_energies"       : concatenated low-lying energies used for summary plots
- "all_charges"        : corresponding normalized Q values
"""
function run_ed_simulation(
    M_val::Int, N::Int, J0::Number, mu::Float64, W::Int;
    krylov_k::Int=8, dense_cutoff::Int=3000, krylov_tol::Float64=1e-9, krylov_maxiter::Int=50_000,
    target_Q_integer::Number=-1, random_J::Bool=false, rng=Random.default_rng()
)
    # 1) Ensure basis exists on all workers + master (build at most once per process)
    @sync for pid in workers()
        @async remotecall_wait(EDModule.ensure_worker_basis!, pid, M_val, N)
    end
    ensure_worker_basis!(M_val, N)

    # 2) Build disorder vector ν and QB tensor J on master
    total_modes = M_val * N
    ν = if W > 0
        Random.seed!(123)                # reproducible disorder
        Float64(W) .* randn(total_modes)
    else
        zeros(total_modes)
    end
    J = generate_Jijkl(N, J0; random_J=random_J, rng=rng)

    # 3) Sector ordering: largest first (helps balance work / expectations)
    B = _basis(M_val, N)
    q_list = collect(keys(B.sectors))
    sort!(q_list, by = q -> -length(B.sectors[q].states))

    # 4a) PASS 1 — parallel: get each sector’s lowest energy (no vectors)
    results = pmap(q_list) do q
        sb = EDModule._sector(M_val, N, q)
        evals = EDModule.sector_lowest_k_eigs(
            sb, M_val, N, J, mu, ν;
            k=1, dense_cutoff=dense_cutoff,
            tol=krylov_tol, maxiter=krylov_maxiter,
            is_target_Q=false,
        )
        (q, isempty(evals) ? +Inf : evals[1])
    end

    # 4b) Choose global minimum sector
    minE = +Inf
    minQ = 0.0
    @inbounds for (q, e0) in results
        if e0 < minE
            minE = e0
            minQ = float(q)
        end
    end

    # 4c) PASS 2 — compute ground-state *vector* in the winning sector (dense or Krylov)
    sb_min  = _sector(M_val, N, minQ)
    dim_min = length(sb_min.states)
    E0 = 0.0
    ψ0 = Vector{Float64}()

    if dim_min <= dense_cutoff
        E0, ψ0 = dense_sector_groundstate(sb_min, M_val, N, J, mu, ν)
    else
        E0, ψ0 = krylov_sector_groundstate(sb_min, M_val, N, J, mu, ν;
                                           tol=krylov_tol, maxiter=krylov_maxiter)
    end

    # Compute densities from the ground-state vector
    per_site, per_mode = densities_from_state(sb_min, ψ0, M_val, N)

    # 5) Optional low-lying energy scatter for summary output / plotting
    all_E = Float64[]
    all_Q = Float64[]
    sector_summaries = pmap(q_list) do q
        sb = EDModule._sector(M_val, N, q)
        evals = EDModule.sector_lowest_k_eigs(
            sb, M_val, N, J, mu, ν;
            k=krylov_k, dense_cutoff=dense_cutoff,
            tol=krylov_tol, maxiter=krylov_maxiter,
            is_target_Q = (q == target_Q_integer),
        )
        (q, evals)
    end
    @inbounds for (q, evals) in sector_summaries
        append!(all_E, evals)
        append!(all_Q, fill(float(q), length(evals)))
    end

    # Normalize and sort the scatter for convenience
    perm = sortperm(all_E)
    all_E_sorted = all_E[perm]
    full_Q0 = B.full_Q0
    all_Q_sorted = (all_Q ./ full_Q0)[perm]

    return Dict(
        "GS_energy"          => E0,
        "GS_charge"          => minQ / full_Q0,
        "GS_site_densities"       => per_site,
        "GS_mode_densities"  => per_mode,
        "H_matrix_dim"       => total_modes,
        "all_energies"       => all_E_sorted,
        "all_charges"        => all_Q_sorted
    )
end

end # module
