function setjinds( j::Integer, sub_len::Integer, full_len::Integer )

    #indices of j-th subvector
    start = Integer((j - 1)*sub_len + 1)
    stop  = Integer(j*sub_len)

    jinds    = collect(start:stop)          # Vector of Integers
    notjinds = setdiff(1:full_len, jinds)   # Vector of Integers

    return (jinds, notjinds)
end

function setjinds(j::Integer, blockinds::Vector)

    jinds      = blockinds[j]

    notjblocks = vcat(1:j-1, j+1:length(blockinds))
    notjinds   = vcat( blockinds[ notjblocks ]... )
    
    return (jinds, notjinds)
end

function blockinds_create(nsz::Integer, nblock::Integer)
    
    # create blockinds
    blockperm = collect(1:nsz)
    blocksz   = div( nsz, nblock )::Integer 

    # blockinds: vector of vectors
    blockinds = Vector{Vector{Integer}}(undef, nblock)

    for i in 1:nblock

        blockinds[i] = blockperm[((i-1)*blocksz + 1):(i*blocksz)]

    end

    return blockinds
end