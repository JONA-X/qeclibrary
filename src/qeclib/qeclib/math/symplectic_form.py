import numpy as np
import itertools
from scipy.special import comb


def commute(op1: np.ndarray | list[int], op2: np.ndarray | list[int]) -> bool:
    """Return whether two pauli operators commute or anticommute.

    Parameters
    ----------
    op1 : np.ndarray | list[int]
        Operator 1 in symplectic vector representation.
    op2 : np.ndarray | list[int]
        Operator 2 in symplectic vector representation.

    Returns
    -------
    bool
        True if the operators commute, False if they anticommute.
    """
    if len(op1) != len(op2):
        raise ValueError("The two operators must have the same length")
    if len(op1) % 2 != 0:
        raise ValueError("The operators must have an even number of elements")

    op1 = np.array(op1)
    op2 = np.array(op2)
    n = op1.shape[0] // 2
    op1_xyz = op1[:n] + 2 * op1[n:]
    op2_xyz = op2[:n] + 2 * op2[n:]

    commute = True
    for i in range(n):
        if (op1_xyz[i] != 0 and op2_xyz[i] != 0) and op1_xyz[i] != op2_xyz[i]:
            commute = not commute

    return commute


def generate_group(vectors: np.ndarray | list[list]) -> np.ndarray:
    """Generate a list of all group elements for the group that is spanned by the given
    vectors. The group elements are represented as symplectic vectors and every element
    corresponds to a row of the returned matrix.

    Parameters
    ----------
    vectors : np.ndarray | list[list]
        Vectors that span the group. They do not need to be linearly independent.

    Returns
    -------
    np.ndarray
        Group spanned by the given vectors. Each row corresponds to a group element.
    """
    vectors = np.array(vectors)  # Accept also list of lists instead of np.ndarray
    m = vectors.shape[0]  # Number of basis vectors
    n = vectors.shape[1] // 2  # Number of qubits

    group_elements = []
    for bitstring in itertools.product("01", repeat=m):
        group_element = np.zeros(2 * n, dtype=int)
        for i, bit in enumerate(bitstring):
            if bit == "1":
                group_element += np.array(vectors[i])
        group_element %= 2
        group_elements.append(group_element)

    # Convert to a set to remove duplicates
    group_elemnts_set = {tuple(element) for element in group_elements}
    return np.array([list(stab) for stab in group_elemnts_set], dtype=int)


def stabilizer_distribution(stab_matrix: np.ndarray) -> np.ndarray:
    """Calculate the first Shor-Laflamme distribution of the stabilizer weights. The
    distribution is a vector `A = (A_0, ..., A_{n+1})` where `A_i` is the number of
    stabilizers with weight `i`. Note that `A_0` is always one since the identity has
    weight zero and is always part of the stabilizer group.

    Parameters
    ----------
    stab_matrix : np.ndarray
        Stabilizer matrix. Every row corresponds to a generator of the stabilizer group.

    Returns
    -------
    np.ndarray
        First Shor-Laflamme distribution of the stabilizer weights.
    """
    n = stab_matrix.shape[1] // 2  # Number of qubits
    distribution = np.zeros(n + 1, dtype=int)
    stabilizer_group = generate_group(stab_matrix)
    for stab in stabilizer_group:
        # Count the number of non-identity paulis in the stabilizer
        # Increase the corresponding entry in the distribution
        distribution[np.sum(stab[:n] + stab[n:] > 0)] += 1
    return distribution


def MacWilliams(n_qubits: int) -> np.ndarray:
    """Return the matrix form of the MacWilliams transform which transforms the first to
    the second Shor-Laflamme distribution. See eq. (7) in
    https://arxiv.org/pdf/2408.16914.

    Parameters
    ----------
    n_qubits : int
        Number of data qubits in the code

    Returns
    -------
    np.ndarray
        MacWilliams transform matrix
    """
    M = np.zeros((n_qubits + 1, n_qubits + 1), dtype=float)
    for i in range(n_qubits + 1):
        for j in range(n_qubits + 1):
            for l in range(i + 1):
                M[i, j] += (
                    comb(n_qubits - j, i - l) * comb(j, l) * (-1) ** l * 3 ** (i - l)
                )
            M[i, j] /= 2**n_qubits
    return M


def normalizer_distribution(
    stab_matrix: np.ndarray = None, stab_distribution: np.ndarray = None
) -> np.ndarray:
    """Calculate the second Shor-Laflamme distribution of the normalizer weights of the
    normalizer N(S) of the given stabilizer group S. The distribution is a vector
    `B = (B_0, ..., B_{n+1})` where `B_i` is the number of elements in the normalizer
    with weight `i`. Note that `B_0` is always one since the identity has weight zero
    and is always part of the normalizer N(S).

    Parameters
    ----------
    stab_matrix : np.ndarray, optional
        Stabilizer matrix. Every row corresponds to a generator of the stabilizer group,
        by default None
    stab_distribution : np.ndarray, optional
        If provided, the first Shor-Laflamme distribution of the stabilizer weights is
        not computed again but instead the given distribution is assumed, by default
        None

    Returns
    -------
    np.ndarray
        Second Shor-Laflamme distribution of the normalizer weights.
    """
    if stab_matrix is not None and stab_distribution is not None:
        raise ValueError(
            "Either stab_matrix or stab_distribution must be provided but not both"
        )

    if stab_distribution is None:
        n = stab_matrix.shape[1] // 2  # Number of qubits
        stab_distribution = stabilizer_distribution(stab_matrix)
    else:
        n = len(stab_distribution) - 1
    distr = MacWilliams(n) @ stab_distribution
    return np.array(distr / distr[0], dtype=int)  # Normalize so that B_0 == 1


def find_distance(stab_matrix: np.ndarray) -> int:
    """Find the code distance of the stabilizer code with the given stabilizer matrix.

    Parameters
    ----------
    stab_matrix : np.ndarray
        Stabilizer matrix. Every row corresponds to a generator of the stabilizer group.

    Returns
    -------
    int
        Code distance of the stabilizer code.
    """
    A = stabilizer_distribution(stab_matrix)
    B = normalizer_distribution(stab_distribution=A)
    return np.where(B - A > 0)[0][0]


def operator_set_commute(operators: np.ndarray | list[list[int]]) -> bool:
    """Check whether the given set of operators commute pairwise.

    Parameters
    ----------
    operators : np.ndarray | list[list[int]]
        List of operators in symplectic vector representation.

    Returns
    -------
    bool
        True if all operators commute with each other respectively, False otherwise.
    """
    for i in range(len(operators)):
        for j in np.arange(i + 1, len(operators)):
            if not commute(operators[i], operators[j]):
                return False
    return True
