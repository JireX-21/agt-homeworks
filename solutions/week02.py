#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def plot_best_response_value_function(row_matrix: np.ndarray, step_size: float) -> None:
    """Plot the best response value function for the row player in a 2xN zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    step_size : float
        The step size for the probability of the first action of the row player
    """
    probabilities = np.arange(0, 1, step_size)
    num_cols = row_matrix.shape[1]
    
    # Calculate expected payoffs for each column action for each probability p
    expected_payoffs = np.array([probabilities * row_matrix[0, j] + (1 - probabilities) * row_matrix[1, j] for j in range(num_cols)])
    
    # The value for the row player for a given p is the minimum of the expected payoffs
    best_response_values = np.min(expected_payoffs, axis=0)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for j in range(num_cols):
        plt.plot(probabilities, expected_payoffs[j], label=f'vs Column Action {j+1}')

    plt.plot(probabilities, best_response_values, 'k--o', linewidth=1, label='Best Response Value (Lower Envelope)')

    plt.xlabel("Probability 'p' of Row Player Choosing Action 1")
    plt.ylabel("Row Player's Expected Payoff")
    plt.title("Row Player's Best Response Value Function")
    plt.legend()
    plt.grid(True)
    plt.show()


def verify_support(
    matrix: np.ndarray, row_support: np.ndarray, col_support: np.ndarray
) -> np.ndarray | None:
    """
    Construct a system of linear equations to check whether there
    exists a candidate for a Nash equilibrium for the given supports.

    The reference implementation uses `scipy.optimize.linprog`
    with the default solver -- 'highs'. You can find more information at
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of one of the players
    row_support : np.ndarray
        The row player's support
    col_support : np.ndarray
        The column player's support

    Returns
    -------
    np.ndarray | None
        The opponent's strategy, if it exists, otherwise `None`
    """
    num_rows, num_cols = matrix.shape

    # Indices of actions in the supports
    row_indices = np.where(row_support == 1)[0]
    col_indices = np.where(col_support == 1)[0]

    # Number of actions in the supports
    m = len(row_indices)
    n = len(col_indices)
    
    if m == 0 or n == 0:
        return None

    # Coefficients for the objective function (minimize -v)
    c = np.zeros(n + 1)
    c[-1] = -1  # We want to maximize v, so we minimize -v

    # Inequality constraints matrix and vector
    A_ub = np.zeros((m, n + 1))
    b_ub = np.zeros(m)

    for i in range(m):
        A_ub[i, :-1] = -matrix[row_indices[i], col_indices]
        A_ub[i, -1] = 1  # Coefficient for v
        b_ub[i] = 0

    # Equality constraints matrix and vector (sum of probabilities = 1)
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :-1] = 1
    b_eq = np.array([1])
    
    # Bounds for each variable (probabilities >= 0 and v is unbounded)
    bounds = [(0, None) for _ in range(n)] + [(None, None)]

    # Solve the linear program
    result = scipy.optimize.linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs'
    )

    if result.success:
        opponent_strategy = result.x[:-1]
        return opponent_strategy
    else:
        return None

import numpy as np
from itertools import combinations

def support_enumeration(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Run Support Enumeration and return all mixed/pure Nash equilibria as
    (row_strategy, col_strategy) pairs of full-length probability vectors.

    Requires a companion function:
        verify_support(matrix, row_support, col_support) -> np.ndarray | None
    which returns the opponent's mixed strategy on the full action set (zeros
    off-support) or None if the support pair is infeasible.
    """
    n_rows, n_cols = row_matrix.shape
    assert col_matrix.shape == (n_rows, n_cols), "Payoff matrices must align."

    eqs: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()
    tol = 1e-8

    # All non-empty supports for rows and columns
    row_supports = [np.array(s, dtype=int)
                    for k in range(1, n_rows + 1)
                    for s in combinations(range(n_rows), k)]
    col_supports = [np.array(s, dtype=int)
                    for k in range(1, n_cols + 1)
                    for s in combinations(range(n_cols), k)]

    for Rs in row_supports:
        for Cs in col_supports:
            # 1) Given (Rs, Cs), find a column mix q that makes rows in Rs indifferent
            q = verify_support(row_matrix, Rs, Cs)
            if q is None:
                continue

            # 2) Symmetric check: find a row mix p that makes columns in Cs indifferent.
            # Reuse verify_support on the transposed game.
            # Here, "rows" are Cs and "cols" are Rs; the returned vector is over original rows.
            p = verify_support(col_matrix.T, Cs, Rs)
            if p is None:
                continue

            # 3) Ensure supports actually match (eliminate spurious zero-prob mixes)
            supp_p = np.flatnonzero(p > tol)
            supp_q = np.flatnonzero(q > tol)
            if not (np.array_equal(np.sort(supp_p), np.sort(Rs)) and
                    np.array_equal(np.sort(supp_q), np.sort(Cs))):
                continue

            # 4) Best-response sanity (redundant but safe): no profitable deviation
            # Row can't gain by deviating:
            row_payoffs = row_matrix @ q
            v_row = float(np.dot(p, row_payoffs))
            if np.any(row_payoffs > v_row + 1e-7):
                continue
            # Column can't gain by deviating:
            col_payoffs = col_matrix.T @ p  # expected payoff for each column action
            v_col = float(np.dot(q, col_payoffs))
            if np.any(col_payoffs > v_col + 1e-7):
                continue

            # 5) Deduplicate (handle the same equilibrium found via different supports)
            key = (tuple(np.round(p, 10)), tuple(np.round(q, 10)))
            if key not in seen:
                seen.add(key)
                eqs.append((p, q))

    return eqs




def main() -> None:
    pass


if __name__ == '__main__':
    main()
