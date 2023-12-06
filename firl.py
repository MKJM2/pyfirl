from time import time as get_time
import numpy as np
import cvxpy as cp
from scipy.sparse import csr_matrix
from mdp import stdpolicy, stdvalueiteration

class TreeNode:
    def __init__(self, type_val, index, test, mean_val, cells=None, ltTree=None, gtTree=None):
        self.type = type_val
        self.index = index
        self.test = test
        self.cells = cells
        self.mean = mean_val
        self.ltTree = ltTree
        self.gtTree = gtTree

def firlmatchdepth(tree, l1, l2) -> int:
    # Check if both leaves match
    if tree.type == 0:
        if tree.index == l1:
            return -1
        elif tree.index == l2:
            return -2
        else:
            return 0
    else:
        mLeft = firlmatchdepth(tree.ltTree, l1, l2)
        mRight = firlmatchdepth(tree.gtTree, l1, l2)
        
        if (mLeft == -1 or mLeft == -2) and mRight == 0:
            return mLeft
        elif (mRight == -1 or mRight == -2) and mLeft == 0:
            return mRight
        elif (mRight == -1 and mLeft == -2) or (mRight == -2 and mLeft == -1):
            return 1
        else:
            matchDepth = max(mLeft, mRight)
            if matchDepth > 0:
                matchDepth += 1
            return matchDepth

# Return index of the leaf that contains state s in tree
def firlcheckleaf(tree, s, feature_data):

    # Check if this is a leaf
    if tree.type == 0:
        # Return result
        return tree.index, tree.mean
    else:
        # Recurse
        if feature_data['splittable'][s, tree.test] == 0:
            branch = tree.ltTree
        else:
            branch = tree.gtTree
        
        return firlcheckleaf(branch, s, feature_data)


def firlaveragereward(tree, R, actions):
    """
    Compute the closest reward function that can be represented by the given tree.

    Args:
    - tree: the tree structure with attributes `type`, `cells`, and `mean`.
    - R: the reward function matrix.
    - actions: the number of actions.

    Returns:
    - Rout: the updated reward function.
    """
    if tree.type == 0:
        count = len(tree.cells)

        # Replace the relevant section of the reward function.
        for i in range(count):
            s = tree.cells[i]
            for a in range(actions):
                R[s][a] = tree.mean[a]
        Rout = R
    else:
        R = firlaveragereward(tree.ltTree, R, actions)
        R = firlaveragereward(tree.gtTree, R, actions)
        Rout = R
    return Rout

def firldefaultparams(algorithm_params={}):
    """
    Fill in default parameters for the FIRL algorithm.

    Args:
    - algorithm_params: dictionary containing provided parameters.

    Returns:
    - algorithm_params: dictionary containing all parameters with defaults filled in.
    """

    # Create default parameters
    default_params = {
        'seed': 0,
        'iterations': 10,
        'depth_step': 1,
        'init_depth': 0
    }
    
    # Set parameters with defaults if not provided
    for key, value in default_params.items():
        algorithm_params.setdefault(key, value)
    
    return algorithm_params

def firlregressiontree(st_states, depth, leavesIn, Eo, R, V, split_thresh, max_depth, mdp_data, feature_data):
    """
    Construct decision subtree.
    """
    leaves = leavesIn
    test = 1
    G = float('inf')

    if depth > max_depth:
        makeLeaf = False
        fMean = R[st_states, :].mean(axis=0)
    else:
        # Step over all possible splitting moves
        for tTest in range(feature_data['splittable'].shape[1]):
            # Split the examples
            st_splits = feature_data['splittable'][st_states, tTest]
            lt_states = st_states[st_splits == 0]
            gt_states = st_states[st_splits == 1]

            # Compute mean
            ltMean = R[lt_states, :].mean()
            gtMean = R[gt_states, :].mean()
            ltVar = ((R[lt_states, :] - ltMean) ** 2).sum()
            gtVar = ((R[gt_states, :] - gtMean) ** 2).sum()
            value = ltVar + gtVar

            if len(lt_states) > 0 and len(gt_states) > 0 and value < G:
                G = value
                test = tTest

        # Construct the partitions
        st_splits = feature_data['splittable'][st_states, test]
        lt_states = st_states[st_splits == 0]
        gt_states = st_states[st_splits == 1]
        fMean = R[st_states, :].mean(axis=0)
        fullMean = fMean.mean()
        maxDeviation = ((R[st_states, :] - fullMean) ** 2).max(axis=0).max()


        if maxDeviation > (split_thresh ** 2) and len(st_states) > 1:
            # Test if this node should be prunable
            Rnew = R.copy()
            Rnew[st_states, :] = fMean
            Vnew = stdvalueiteration(mdp_data, Rnew, V)
            _, P = stdpolicy(mdp_data, Rnew, Vnew)

            # Test if P matches all non-zero values of Eo
            mismatches = Eo * (P != Eo)
            makeLeaf = len(np.nonzero(mismatches)[0]) != 0
        else:
            makeLeaf = False

    if makeLeaf and len(st_states) > 1 and G != float('inf'):
        # Create node with the best split
        rightTree, leaves, R, V = firlregressiontree(gt_states, depth+1, leaves, Eo, R, V, split_thresh, max_depth, mdp_data, feature_data)
        leftTree, leaves, R, V = firlregressiontree(lt_states, depth+1, leaves, Eo, R, V, split_thresh, max_depth, mdp_data, feature_data)
        
        # Create node. TreeNode constructor parameters:
        # type_val, index, test, mean_val, cells=None, ltTree=None, gtTree=None
        tree = TreeNode(1, None, test, fMean, st_states.tolist(), leftTree, rightTree)
        Rout = R
        Vout = V
    else:
        # Create leaf node
        # tree = {'type': 0, 'index': leaves + 1, 'mean': fMean, 'cells': st_states.tolist()}
        tree = TreeNode(0, leaves, None, fMean, st_states.tolist())
        leaves += 1
        Rout = R
        Vout = V

    return tree, leaves, Rout, Vout


def firloptimization(Eo, Rold, ProjToLeaf, LeafToProj, FeatureMatch, mdp_data, verbosity):
    """
    Runs the optimization phase to compute a reward function that is close to 
    the current feature hypothesis 
    """

    # Smoothing term (relative to reward objective)
    # SMOOTH_WEIGHT = 0.02
    SMOOTH_WEIGHT = 0.001

    # Total size
    states = mdp_data['states']
    actions = mdp_data['actions']
    msize = states * actions
    results = mdp_data['sa_s'].shape[2]

    ### Constraint construction ###
    cols = np.nonzero(Eo)[0]
    examples = len(cols)

    sN = np.zeros(msize - examples * actions, dtype=int)            # start state idxs
    rN = np.zeros(msize - examples * actions, dtype=int)            # state-action idxs
    eN = np.zeros((msize - examples * actions, results), dtype=int) # resultant state idxs
    pN = np.zeros((msize - examples * actions, results))            # resultant state coeffs

    sM = np.zeros(examples * (actions - 1), dtype=int)
    rM = np.zeros(examples * (actions - 1), dtype=int)
    eM = np.zeros((examples * (actions - 1), results), dtype=int)
    pM = np.zeros((examples * (actions - 1), results))

    sE = np.zeros(examples, dtype=int)
    rE = np.zeros(examples, dtype=int)
    eE = np.zeros((examples, results), dtype=int)
    pE = np.zeros((examples, results))

    Nrow = 0
    Mrow = 0
    Erow = 0
    for startstate in range(states):
        if Eo[startstate] != 0:
            # We generate destination state and reward under the optimal action
            optaction = Eo[startstate]
            reward = actions * startstate + optaction

            sE[Erow] = startstate
            rE[Erow] = reward
            eE[Erow, :] = mdp_data['sa_s'][startstate, optaction, :]
            pE[Erow, :] = mdp_data['sa_p'][startstate, optaction, :] * mdp_data['discount']
            Erow += 1

            for action in range(actions):
                if action != optaction:
                    reward = actions * startstate + action

                    sM[Mrow] = startstate
                    rM[Mrow] = reward
                    eM[Mrow, :] = mdp_data['sa_s'][startstate, action, :]
                    pM[Mrow, :] = mdp_data['sa_p'][startstate, action, :] * mdp_data['discount']
                    Mrow += 1
        else:
            for action in range(actions):
                # Generate destination state and reward indices
                reward = actions * startstate + action

                sN[Nrow] = startstate
                rN[Nrow] = reward
                eN[Nrow, :] = mdp_data['sa_s'][startstate, action, :]
                pN[Nrow, :] = mdp_data['sa_p'][startstate, action, :] * mdp_data['discount']
                Nrow += 1

    # Determine number of leaves
    _, msize = ProjToLeaf.shape
    leafEntries, leaves = FeatureMatch.shape

    # Margin by which examples should be optimal
    MARGIN = 0.01
    margins = np.ones(examples * (actions - 1)) * MARGIN

    EPSILON = 2.22e-16
    r = cp.Variable(msize)
    v = cp.Variable(states)
    f = cp.Variable(leaves)

    objective = cp.Minimize(cp.norm(LeafToProj @ f - r) ** 2 / msize +
        cp.norm(FeatureMatch @ f, 1) * (SMOOTH_WEIGHT / (leafEntries * 500)))

    constraints = [
        f == ProjToLeaf @ r,
        #v[sN] >= r[rN] + cp.sum(cp.multiply(v[eN], pN), axis=1),
        #v[sM] >= r[rM] + cp.sum(cp.multiply(v[eM],pM), axis=1) + margins,
        #v[sE] == r[rE] + cp.sum(cp.multiply(v[eE], pE), axis=1)
    ]

    # NOTE: In CVXPY, we can't index a variable directly with a list or array
    # of indices like we can in MATLAB. Hence:

    # Add constraints for sN, rN, eN, and pN
    for idx in range(len(sN)):
        constraints.append(v[sN[idx]] >= r[rN[idx]] +
                           cp.sum(cp.multiply(v[eN[idx, :]], pN[idx, :])))

    # Add constraints for sM, rM, eM, and pM with margins
    for idx in range(len(sM)):
        constraints.append(v[sM[idx]] >= r[rM[idx]] +
                           cp.sum(cp.multiply(v[eM[idx, :]], pM[idx, :])) + margins[idx])

    # Add constraints for sE, rE, eE, and pE
    for idx in range(len(sE)):
        constraints.append(v[sE[idx]] == r[rE[idx]] +
                           cp.sum(cp.multiply(v[eE[idx, :]], pE[idx, :])))

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbosity == 2)

    if Rold.shape[0] > 1 and np.isnan(prob.value):
        if verbosity != 0:
            print('WARNING: Failed to obtain solution, reverting to old reward!')
        R = Rold
    else:
        # Recover the reward function
        R = r.value.reshape(actions, states).T

    return R, MARGIN



def firlprojectionfromtree(tree, leaves, states, actions, feature_data):
    
    DEPTH_WEIGHT = 1
    
    # Matrix of adjacencies.
    adjleaves = csr_matrix((leaves, leaves), dtype=np.int32)
    stateleaves = np.zeros(states, dtype=np.int32)
    
    # Count number of elements in each leaf and assign leaf to each state.
    elements = np.zeros(leaves, dtype=np.int32)
    for s in range(states):
        leaf, _mean = firlcheckleaf(tree, s, feature_data)
        elements[leaf] += 1
        stateleaves[s] = leaf
        
    # Count pairs and build adjacency matrix.
    pairs = 0
    for s in range(states):
        leaf = stateleaves[s]
        adj = np.nonzero(feature_data['stateadjacency'][s, :])[0]
        numadj = len(adj)
        
        # Write out adjacencies
        for i in adj:
            lother = stateleaves[i]
            if lother != leaf:
                # Found adjacency
                if adjleaves[lother, leaf] == 0 and adjleaves[leaf, lother] == 0:
                    pairs += 1
                adjleaves[lother, leaf] = 1
                adjleaves[leaf, lother] = 1
                
    # Construct feature match matrix
    FeatureMatch = csr_matrix((pairs, leaves), dtype=np.float64)
    idx = 0
    maxPair = 0
    for l1 in range(leaves):
        for l2 in range(l1 + 1, leaves):
            adjacent = adjleaves[l1, l2]
            if adjacent > 0:
                matchDepth = (firlmatchdepth(tree, l1, l2) - 1)
                FeatureMatch[idx, l1] = adjacent + matchDepth * DEPTH_WEIGHT
                FeatureMatch[idx, l2] = -adjacent - matchDepth * DEPTH_WEIGHT
                if FeatureMatch[idx, l1] > maxPair:
                    maxPair = FeatureMatch[idx, l1]
                idx += 1
    
    if pairs <= 0:
        # Handle degeneracy
        FeatureMatch = csr_matrix((1, leaves), dtype=np.float64)
    else:
        FeatureMatch = FeatureMatch / maxPair
    
    # Construct projection matrix
    ProjToLeaf = csr_matrix((leaves, states * actions), dtype=np.float64)
    LeafToProj = csr_matrix((states * actions, leaves), dtype=np.float64)
    
    for s in range(states):
        leaf = stateleaves[s]
        for a in range(actions):
            pos = s * actions + a
            ProjToLeaf[leaf, pos] = 1.0 / (elements[leaf] * actions)
            LeafToProj[pos, leaf] = 1.0

    # Convert to CSR for efficient operations in future usage
    return ProjToLeaf.tocsr(), LeafToProj.tocsr(), FeatureMatch.tocsr()


def firlrun(algorithm_params, mdp_data, mdp_model, feature_data, example_samples, _, verbosity):

    # Fill in default parameters
    algorithm_params = firldefaultparams(algorithm_params)

    np.random.seed(algorithm_params['seed'])

    # Initialize variables
    states = mdp_data['states']
    actions = mdp_data['actions']
    iterations = algorithm_params['iterations']
    depth_step = algorithm_params['depth_step']
    init_depth = algorithm_params['init_depth']

    # Construct mapping from states to example actions
    Eo = np.zeros(states, dtype=int)
    for i in range(len(example_samples)):
        for t in range(len(example_samples[i])):
            Eo[example_samples[i][t][0]] = example_samples[i][t][1]

    # Construct initial tree
    leaves = 1
    # Note: In python should be zero indexed
    # tree = {'type': 0, 'index': 0, 'mean': np.zeros(actions)}
    tree = TreeNode(0, 0, None, np.zeros(actions))
    ProjToLeaf, LeafToProj, FeatureMatch = firlprojectionfromtree(tree, leaves, states, actions, feature_data)

    # Prepare timing variables.
    optTime, fitTime, vitTime, matTime = [np.zeros(iterations) for _ in range(4)]
    
    # Prepare intermediate output variables
    opt_acc_itr = [None] * (iterations)
    r_itr = [None] * (iterations)
    p_itr = [None] * (iterations)
    model_itr = [None] * (iterations)
    model_r_itr = [None] * (iterations)
    model_p_itr = [None] * (iterations)
    
    # Run firl.
    Rold = np.random.normal(size=(states, actions))
    itr = 0
    while True:
        if verbosity != 0:
            print(f'Beginning FIRL iteration {itr+1}')

        # Run optimization phase
        start_time = get_time()
        R, margin = firloptimization(Eo, Rold, ProjToLeaf, LeafToProj, FeatureMatch, mdp_data, verbosity)
        Rold = R
        threshold = margin * 0.2 * mdp_data['discount']
        optTime[itr] = get_time() - start_time

        # Generate policy
        start_time = get_time()
        V = stdvalueiteration(mdp_data, R)
        _, P = stdpolicy(mdp_data, R, V)
        vitTime[itr] = get_time() - start_time

        # Construct tree
        start_time = get_time()
        # Adjust Eo to exclude violated examples
        # In an exact optimization, there should be no violated examples
        # However, an approximation might violate some examples
        Eadjusted = Eo * (P == Eo)
        totalExamples = np.sum(Eadjusted > 0)
        #opt_acc_itr.append(totalExamples / np.sum(Eo > 0))
        opt_acc_itr[itr] = totalExamples / np.sum(Eo > 0)
        max_depth = init_depth + itr * depth_step
        tree, leaves, _, _ = firlregressiontree(
            np.arange(states),      # Start with all states
            0,                      # Current depth
            0,                      # First leaf index
            Eadjusted,              # Pass in part of policy we want to match
            R,                      # Pass in reward function
            V,                      # Pass in value function
            threshold,              # Pass in termination threshold
            max_depth,              # Pass in maximum depth
            mdp_data,               # Pass in MDP data
            feature_data            # Pass in feature data
        )
        fitTime[itr] = get_time() - start_time

        # Construct projection matrices
        start_time = get_time()
        ProjToLeaf, LeafToProj, FeatureMatch = firlprojectionfromtree(tree, leaves, states, actions, feature_data)
        matTime[itr] = get_time() - start_time

        # Record policy at this iteration
        #r_itr.append(R)
        #p_itr.append(P)
        #model_itr.append(tree)
        r_itr[itr] = R
        p_itr[itr] = P
        model_itr[itr] = tree

        # Increment iteration
        itr += 1
        if itr >= iterations:
            break

    # Compute final policy
    Rout = firlaveragereward(tree, R, actions)
    Vout = stdvalueiteration(mdp_data, Rout)
    Qout, Pout = stdpolicy(mdp_data, Rout, Vout)

    # Compute all intermediate policies
    for i in range(iterations):
        model_r_itr[i] = firlaveragereward(model_itr[i], r_itr[i], actions)
        v = stdvalueiteration(mdp_data, model_r_itr[i])
        _, model_p_itr[i] = stdpolicy(mdp_data, model_r_itr[i], v)

    if verbosity != 0:
        # Report timing
        for itr in range(iterations):
            print(f'Iteration {itr + 1} optimization: {optTime[itr]:.6f}s')
            print(f'Iteration {itr + 1} value iteration: {vitTime[itr]:.6f}s')
            print(f'Iteration {itr + 1} fitting: {fitTime[itr]:.6f}s')
            print(f'Iteration {itr + 1} objective construction: {matTime[itr]:.6f}s')

    total = sum(optTime) + sum(vitTime) + sum(fitTime) + sum(matTime)
    if verbosity != 0:
        print(f'Total time: {total:.6f}s\n')

    time = total
    mean_opt_time = np.mean(optTime)
    mean_fit_time = np.mean(fitTime)

    # Build output structure
    irl_result = {
        'r': Rout,
        'v': Vout,
        'q': Qout,
        'p': Pout,
        'opt_acc_itr': opt_acc_itr,
        'r_itr': r_itr,
        'model_itr': model_itr,
        'model_r_itr': model_r_itr,
        'p_itr': p_itr,
        'model_p_itr': model_p_itr,
        'time': time,
        'mean_opt_time': mean_opt_time,
        'mean_fit_time': mean_fit_time
    }

    return irl_result
