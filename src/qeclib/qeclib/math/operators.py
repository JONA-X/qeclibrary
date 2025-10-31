def sim_readout(op1: list[int], op2: list[int]) -> bool:
    """Check whether the two operators can be read out simultaneously.

    Parameters
    ----------
    op1 : list[int]
        The first operator in F2 representation.
    op2 : list[int]
        The second operator in F2 representation.

    Returns
    -------
    bool
        True if the operators can be read out simultaneously, False otherwise.
    """
    n = len(op1) // 2
    for i in range(n):
        if (op1[i] == 1 or op1[i+n] == 1) and (op2[i] == 1 or op2[i+n] == 1):
            if op2[i] != op1[i] or op2[i+n] != op1[i+n]:
                return False
    return True
