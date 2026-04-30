import numpy as np

def entropy(p):
    """
    Calculate the Shannon entropy of a probability distribution.
    
    Parameters:
    -----------
    p : array-like
        Probability distribution
        
    Returns:
    --------
    float
        Shannon entropy value
    """
    # Filter out zero probabilities to avoid log(0)
    p = p[p > 0]
    # Calculate entropy using the formula: -sum(p * log(p))
    return -np.sum(p * np.log(p))


def entropic_bipartite(X, max_iter=100, tol=1e-8):
    """
    Perform entropic bipartite clustering on a matrix X.
    
    Parameters:
    -----------
    X : array-like
        Input matrix to cluster
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    tol : float, optional
        Convergence tolerance (default: 1e-8)
        
    Returns:
    --------
    tuple
        (H_rows, H_cols) - Entropy values for rows and columns
    """
    # Convert input to float numpy array
    X = np.array(X, dtype=float)
    N, M = X.shape

    # Calculate row and column sums
    row_sums = X.sum(axis=1, keepdims=True)
    col_sums = X.sum(axis=0, keepdims=True)

    # Initialize row and column probability distributions
    # Avoid division by zero by replacing zeros with ones in denominators
    xi = X / np.where(row_sums == 0, 1, row_sums)
    zeta = X / np.where(col_sums == 0, 1, col_sums)

    # Calculate initial entropy for each row and column
    H_rows = np.array([entropy(xi[i]) for i in range(N)])
    H_cols = np.array([entropy(zeta[:, j]) for j in range(M)])

    # Iterative optimization
    for k in range(max_iter):
        # Store previous entropy values to check convergence
        H_rows_old = H_rows.copy()
        H_cols_old = H_cols.copy()

        # Calculate weighting factors based on current entropy values
        f = np.log(N) - H_cols
        g = np.log(M) - H_rows

        # Ensure numerical stability by setting minimum values
        f = np.maximum(f, 1e-12)
        g = np.maximum(g, 1e-12)

        # Update row probability distributions
        xi_new = np.zeros_like(X)
        for i in range(N):
            weights = X[i] * f
            s = weights.sum()
            if s > 0:
                xi_new[i] = weights / s

        # Update column probability distributions
        zeta_new = np.zeros_like(X)
        for j in range(M):
            weights = X[:, j] * g
            s = weights.sum()
            if s > 0:
                zeta_new[:, j] = weights / s

        # Recalculate entropy values with updated distributions
        H_rows = np.array([entropy(xi_new[i]) for i in range(N)])
        H_cols = np.array([entropy(zeta_new[:, j]) for j in range(M)])

        # Check for convergence using Euclidean norm of differences
        diff = np.linalg.norm(H_rows - H_rows_old) + np.linalg.norm(H_cols - H_cols_old)

        # Break if converged
        if diff < tol:
            print(f"Converged in {k} iterations")
            break

    return H_rows, H_cols