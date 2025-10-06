#!/usr/bin/env python3

import numpy as np


def evaluate_general_sum(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    """Compute the expected utility of each player in a general-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """
    row_utility = row_strategy @ row_matrix @ col_strategy
    col_utility = row_strategy @ col_matrix @ col_strategy

    return np.array([row_utility, col_utility])


def evaluate_zero_sum(
    row_matrix: np.ndarray, row_strategy: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    """Compute the expected utility of each player in a zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """
    utility = row_strategy @ row_matrix @ col_strategy

    return np.array([utility, -utility])

def calculate_best_response_against_row(
    col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.ndarray:
    """Compute a pure best response for the column player against the row player.

    Parameters
    ----------
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy

    Returns
    -------
    np.ndarray
        The column player's best response
    """
    col_payoffs = row_strategy @ col_matrix
    best_response_action = np.argmax(col_payoffs)
    best_response_strategy = np.zeros_like(col_payoffs)
    best_response_strategy[best_response_action] = 1.0
    return best_response_strategy



def calculate_best_response_against_col(
    row_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    """Compute a pure best response for the row player against the column player.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        The row player's best response
    """
    col_payoffs = row_matrix @ col_strategy
    best_response_action = np.argmax(col_payoffs)
    best_response_strategy = np.zeros_like(col_payoffs)
    best_response_strategy[best_response_action] = 1.0
    return best_response_strategy


def evaluate_row_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.float32:
    """Compute the utility of the row player when playing against a best response strategy.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy

    Returns
    -------
    np.float32
        The expected utility of the row player
    """
    col_strategy = calculate_best_response_against_row(col_matrix, row_strategy)
    return row_strategy @ row_matrix @ col_strategy


def evaluate_col_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.float32:
    """Compute the utility of the column player when playing against a best response strategy.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float32
        The expected utility of the column player
    """
    row_strategy = calculate_best_response_against_col(row_matrix, col_strategy)
    return row_strategy @ col_matrix @ col_strategy


def find_strictly_dominated_actions(matrix: np.ndarray) -> np.ndarray:
    """Find strictly dominated actions for the given normal-form game.

    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of one of the players

    Returns
    -------
    np.ndarray
        Indices of strictly dominated actions
    """
    num_actions = matrix.shape[0]
    dominated_actions = set()
    for i in range(num_actions):
        for j in range(num_actions):
            if i != j and np.all(matrix[i] < matrix[j]):
                dominated_actions.add(i)
    return np.array(sorted(dominated_actions))


def iterated_removal_of_dominated_strategies(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Iterated Removal of Dominated Strategies.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Four-tuple of reduced row and column payoff matrices, and remaining row and column actions
    """
    row_actions = np.arange(row_matrix.shape[0])
    col_actions = np.arange(col_matrix.shape[1])

    while True:
        row_dominated = find_strictly_dominated_actions(row_matrix)
        col_dominated = find_strictly_dominated_actions(np.transpose(col_matrix))

        if len(row_dominated) == 0 and len(col_dominated) == 0:
            break

        if len(row_dominated) > 0:
            row_matrix = np.delete(row_matrix, row_dominated, axis=0)
            col_matrix = np.delete(col_matrix, row_dominated, axis=0)
            row_actions = np.delete(row_actions, row_dominated)

        if len(col_dominated) > 0:
            row_matrix = np.delete(row_matrix, col_dominated, axis=1)
            col_matrix = np.delete(col_matrix, col_dominated, axis=1)
            col_actions = np.delete(col_actions, col_dominated)

    return row_matrix, col_matrix, row_actions, col_actions


def main() -> None:
    pass


if __name__ == '__main__':
    main()
