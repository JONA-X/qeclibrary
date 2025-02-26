import numpy as np
import itertools
from scipy.special import comb
import warnings


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
    distribution is a vector `A = (A_0, ..., A_n)` where `A_i` is the number of
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
    `B = (B_0, ..., B_n)` where `B_i` is the number of elements in the normalizer
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


def is_valid_tableau(
    stabilizers: np.ndarray,
    logical_operators: list[tuple[tuple[int, ...], tuple[int, ...]]],
) -> bool:
    """Check whether the given stabilizer tableau is valid, i.e. whether it satisfies
    the following conditions:
    - All stabilizers commute with each other
    - All logical operators commute with all stabilizers
    - For every pair (X_Li, Z_Li) of logical operators, X_Li and Z_Li anticommute
    - All logical operators commute with each other, except for other operator of the
    respective pair. I.e. X_Li and Z_Li commute with every X_Lj and Z_Lj for j != i

    Parameters
    ----------
    stabilizers : np.ndarray
        Stabilizer matrix where every row corresponds to a stabilizer generator in the
        symplectic vector representation.
    logical_operators : list[tuple[tuple[int, ...], tuple[int, ...]]]
        List of logical operator pairs where every tuple contains the X and Z operator
        in symplectic vector representation.

    Returns
    -------
    bool
        True if the tableau is valid, False otherwise.

    Raises
    ------
    Warning
        Print a warning if the system is not fully defined, i.e. if m + k != n
    """
    # Check that all stabilizers commute with each other
    if not operator_set_commute(stabilizers):
        warnings.warn("The stabilizers do not commute.")
        return False

    # Check for every logical operator:
    for i in range(len(logical_operators)):
        # Check that the logical operators commute with the stabilizers
        if not operator_set_commute(
            np.vstack((stabilizers, logical_operators[i][0]))
        ) or not operator_set_commute(
            np.vstack((stabilizers, logical_operators[i][1]))
        ):
            warnings.warn("The logical operators do not commute with all stabilizers.")
            return False

        # Check that the logical operator pair anticommutes
        if commute(logical_operators[i][0], logical_operators[i][1]):
            warnings.warn(
                "The pairs of logical operators do not anticommute respectively."
            )
            return False

        # Check that they commute with all other logical operators
        for j in np.arange(i + 1, len(logical_operators)):
            if (
                not commute(logical_operators[i][0], logical_operators[j][0])
                or not commute(logical_operators[i][0], logical_operators[j][1])
                or not commute(logical_operators[i][1], logical_operators[j][0])
                or not commute(logical_operators[i][1], logical_operators[j][1])
            ):
                warnings.warn(
                    "The logical operators do not commute with the logical operators of other pairs."
                )
                return False

    # Check that the system is fully defined
    n_qubits = stabilizers.shape[1] // 2
    warnings.simplefilter("always", UserWarning)
    if (
        np.linalg.matrix_rank(
            np.vstack(
                (
                    stabilizers,
                    [logical_operators[i][0] for i in range(len(logical_operators))],
                )
            )
        )
        < n_qubits
    ):
        warnings.warn("The system is not fully defined, i.e. m + k < n")
    if len(stabilizers) + len(logical_operators) > n_qubits:
        warnings.warn(
            "The system is over-defined, i.e. m + k > n. The stabilizers or logical operators are not independent."
        )

    return True


def cartesian_product_of_sets(set1: np.ndarray, set2: np.ndarray) -> np.ndarray:
    """Calculates the Cartesian product of two sets in binary symplectic
    representation. I.e. return all possible combinations of elements from the two sets.

    Parameters
    ----------
    set1 : np.ndarray
        Set 1, expressed as a matrix where the rows are the elements of the set in
        binary symplectic representation.
    set2 : np.ndarray
        Set 2, expressed as a matrix where the rows are the elements of the set in
        binary symplectic representation.

    Returns
    -------
    np.ndarray
        Cartesian product of the two sets, expressed as a matrix where the rows are
        the elements of the set in binary symplectic representation.
    """
    # Note: Use a set to avoid duplicates
    # This requires elements of the set to be hashable, therefore use tuples instead of
    # lists or numpy arrays
    return {tuple((el1 + el2) % 2) for el1 in set1 for el2 in set2}


def get_log_op_distribution(
    log_ops: np.ndarray, stabilizer_matrix: np.ndarray
) -> list[int]:
    """Get the weight distribution of a given logical operator group with respect to a
    stabilizer group. The weight distribution is a vector `B = (B_0, ..., B_n)` where
    `B_i` is the number of logical operators with weight `i`.

    Parameters
    ----------
    log_ops : np.ndarray
        Every row corresponds to a logical operator in symplectic vector representation.
    stabilizer_matrix : np.ndarray
        Every row corresponds to a stabilizer in symplectic vector representation. It's
        enough to provide the generators of the stabilizer group.

    Returns
    -------
    list[int]
        Weight distribution of the logical operators
    """
    stabilizer_matrix = np.array(stabilizer_matrix)
    n = len(stabilizer_matrix[0]) // 2  # Number of qubits
    all_ops = cartesian_product_of_sets(
        generate_group(log_ops), generate_group(stabilizer_matrix)
    )
    weights = [np.sum(np.array(op[:n]) + np.array(op[n:]) > 0) for op in all_ops]
    distribution = np.zeros(n + 1, dtype=int)
    for weight in weights:
        distribution[weight] += 1
    return list(distribution)


def F2_to_xyz(operator: list[int]) -> str:
    """Convert a binary symplectic vector representation over F2 of a Pauli operator to
    a string representation containing the Pauli operators X, Y, Z, and I.

    Parameters
    ----------
    operator : list[int]
        Pauli operator in binary symplectic vector representation over F2.

    Returns
    -------
    str
        String representation of the Pauli operator.
    """
    n = len(operator) // 2  # Number of qubits
    str_rep = ""
    for i in range(n):
        if operator[i] == 1 and operator[i + n] == 0:
            str_rep += "X"
        elif operator[i] == 0 and operator[i + n] == 1:
            str_rep += "Z"
        elif operator[i] == 1 and operator[i + n] == 1:
            str_rep += "Y"
        else:
            str_rep += "I"
    return str_rep
