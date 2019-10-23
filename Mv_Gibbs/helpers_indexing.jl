function setjinds( j::Integer, sub_len::Integer, full_len::Integer )

    #indices of j-th subvector
    start = Integer((j - 1)*sub_len + 1)
    stop  = Integer(j*sub_len)

    jinds    = start:stop                   # UnitRange
    notjinds = setdiff(1:full_len, jinds)   # Vector of Integers

    return (jinds, notjinds)
end