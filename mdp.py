import numpy as np

def stdvalueiteration(mdp_data, r, vinit=None):
    """
    Run value iteration to solve a standard MDP.

    Parameters:
        mdp_data (dict): Contains MDP related data.
        r (numpy array): Reward function.
        vinit (numpy array, optional): Initial value function.

    Returns:
        numpy array: Computed value function.
    """
#     print(f"mdp_data: {mdp_data}")
#     print(f"r: {r}")
#     print(f"vinit: {vinit}")
    
    # Allocate initial value function & variables.
    diff = 1.0
    if vinit is not None:
        vn = vinit
    else:
        vn = np.zeros(mdp_data['states'])

    # Perform value iteration.
    while diff >= 1e-8:  # Using 1e-8 as the convergence threshold
        vp = vn
        vn = np.max(r + np.sum(mdp_data['sa_p'] * vp[mdp_data['sa_s']], axis=2) * mdp_data['discount'], axis=1)
        diff = np.max(np.abs(vn - vp))

    # Return value function.
    return vn


def stdpolicy(mdp_data, r, v):
    """
    Given reward and value functions, solve for q function and policy.

    Parameters:
        mdp_data (dict): Contains MDP related data.
        r (numpy array): Reward function.
        v (numpy array): Value function.

    Returns:
        tuple: q function and policy.
    """

    # Compute Q function.
    q = r + np.sum(mdp_data['sa_p'] * v[mdp_data['sa_s']], axis=2) * mdp_data['discount']

    # Compute policy.
    p = np.argmax(q, axis=1)

    return q, p
