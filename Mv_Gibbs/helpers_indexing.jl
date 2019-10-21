function setjinds( j::Int, sub_len::Int, full_len::Int )

    #indices of j-th subvector
    start = Int((j - 1)*sub_len + 1)
    stop  = Int(j*sub_len)

    jinds    = start:stop                   # UnitRange
    notjinds = setdiff(1:full_len, jinds)   # Vector of Ints

    return (jinds, notjinds)
end