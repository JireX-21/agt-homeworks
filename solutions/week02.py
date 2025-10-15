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

    # Get the indices of the actions that are in the supports
    row_indices = np.where(row_support)[0]
    col_indices = np.where(col_support)[0]

    # Get the number of actions in each support
    num_row_support = len(row_indices)
    num_col_support = len(col_indices)

    # If either support is empty, no solution exists
    if num_row_support == 0 or num_col_support == 0:
        return None

    # --- Set up the Linear Program ---
    # We want to find the opponent's strategy (q) and the player's utility (v).
    # The variables for the LP will be the probabilities for the opponent's
    # supported actions, followed by the utility v.
    # x = [q_1, q_2, ..., q_n, v], where n = num_col_support

    # 1. Objective function: We are solving a feasibility problem, but we can
    # frame it as maximizing the utility 'v', which is equivalent to minimizing '-v'.
    # The coefficients 'c' correspond to the variables [q_1, ..., q_n, v].
    c = np.zeros(num_col_support + 1)
    c[-1] = -1  # Coefficient for v is -1 to minimize -v

    # 2. Equality constraints (A_eq * x = b_eq):
    # - The sum of the opponent's probabilities must equal 1.
    # - The player must be indifferent between all actions in their support.
    #   (Payoff for each supported row action must equal v).
    A_eq = np.zeros((num_row_support + 1, num_col_support + 1))
    b_eq = np.zeros(num_row_support + 1)

    # Constraint: sum(q_j) = 1
    A_eq[0, :-1] = 1
    b_eq[0] = 1

    # Constraints: A[i,:] * q - v = 0 for all i in row_support
    sub_matrix = matrix[row_indices][:, col_indices]
    A_eq[1:, :-1] = sub_matrix
    A_eq[1:, -1] = -1  # Subtract v

    # 3. Inequality constraints (A_ub * x <= b_ub):
    # - The player's payoff for any action *not* in their support must be
    #   less than or equal to v.
    rows_not_in_support = np.where(~row_support)[0]
    num_rows_not_in_support = len(rows_not_in_support)

    if num_rows_not_in_support > 0:
        A_ub = np.zeros((num_rows_not_in_support, num_col_support + 1))
        b_ub = np.zeros(num_rows_not_in_support)
        
        # Constraints: A[k,:] * q - v <= 0 for all k not in row_support
        sub_matrix_not_supported = matrix[rows_not_in_support][:, col_indices]
        A_ub[:, :-1] = sub_matrix_not_supported
        A_ub[:, -1] = -1 # Subtract v
    else:
        # No inequality constraints if all rows are in the support
        A_ub = None
        b_ub = None

    # 4. Bounds for variables:
    # - Probabilities (q_j) must be non-negative (>= 0).
    # - Utility (v) is unbounded.
    bounds = [(0, None) for _ in range(num_col_support)] + [(None, None)]

    # Solve the linear program
    res = scipy.optimize.linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs'
    )

    # If the solver found a solution, return the opponent's strategy
    if res.success:
        # The solution for the opponent's strategy (q)
        opponent_strategy_on_support = res.x[:-1]
        
        # Create the full strategy vector, with zeros for non-supported actions
        full_opponent_strategy = np.zeros(num_cols)
        full_opponent_strategy[col_indices] = opponent_strategy_on_support
        return full_opponent_strategy
    else:
        # If no solution was found, the supports are not part of a NE
        return None
    
from itertools import product

def support_enumeration(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run the Support Enumeration algorithm and return a list of all Nash equilibria

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list of strategy profiles corresponding to found Nash equilibria
    """
    num_rows, num_cols = row_matrix.shape
    equilibria = []

    # Generate all non-empty subsets of row and column indices
    row_subsets = [
        np.array([1 if i in s else 0 for i in range(num_rows)], dtype=bool)
        for r in range(1, num_rows + 1)
        for s in __import__("itertools").combinations(range(num_rows), r)
    ]
    col_subsets = [
        np.array([1 if i in s else 0 for i in range(num_cols)], dtype=bool)
        for r in range(1, num_cols + 1)
        for s in __import__("itertools").combinations(range(num_cols), r)
    ]

    # Iterate over all pairs of supports
    for row_support, col_support in product(row_subsets, col_subsets):
        # Find a candidate strategy for the row player that makes the column player indifferent
        row_strategy = verify_support(col_matrix.T, col_support, row_support)

        # Find a candidate strategy for the column player that makes the row player indifferent
        col_strategy = verify_support(row_matrix, row_support, col_support)

        # If both strategies exist, we have a potential Nash Equilibrium
        if row_strategy is not None and col_strategy is not None:
            # Final check: The returned strategies must have supports that exactly match
            # the supports we are currently testing. A probability of 0 in the returned
            # strategy for an action we assumed was in the support means our assumption was wrong.
            # We use a small tolerance for floating point comparisons.
            TOL = 1e-6
            is_row_support_correct = np.all(row_strategy[row_support] > TOL)
            is_col_support_correct = np.all(col_strategy[col_support] > TOL)

            if is_row_support_correct and is_col_support_correct:
                equilibria.append((row_strategy, col_strategy))

    return equilibria


def main() -> None:
    pass


if __name__ == '__main__':
    main()
