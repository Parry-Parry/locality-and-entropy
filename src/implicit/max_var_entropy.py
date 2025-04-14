import math
import copy

def entropy_func(scores):
    """
    Given a list of scores, compute the entropy after normalizing these scores to form
    a probability distribution. If the sum of scores is zero, returns 0.
    """
    total = sum(scores)
    if total == 0:
        return 0.0
    norm_scores = [s / total for s in scores]
    return -sum(p * math.log(p) for p in norm_scores if p > 0)

def compute_entropy_for_subranking(subranking, score_lookup, query_id):
    """
    Given a subranking (list of doc_ids) for a given query_id,
    uses the score_lookup dictionary to compute the entropy.
    """
    scores = [score_lookup[query_id].get(doc_id, 0) for doc_id in subranking]
    return entropy_func(scores)

def optimize_subrankings_pointwise(total_rankings, score_lookup, c, max_iter=100, tol=1e-5):
    """
    total_rankings : dict
       A mapping {query_id: List[doc_id]} representing the full ranking for each query.
    score_lookup : dict
       A mapping {query_id: {doc_id: score}} to lookup scores of docs.
    c : int
       The size of the desired sub-ranking.
    max_iter : int
       Maximum number of iterations for the greedy coordinate update.
    tol : float
       Tolerance for global variance improvement.

    Returns:
       A dictionary {query_id: List[doc_id]} representing the selected sub-rankings.
    """
    # Number of queries.
    query_ids = list(total_rankings.keys())
    num_queries = len(query_ids)

    # INITIALIZATION: For each query, choose the first c documents as the initial subranking.
    subrankings = {}
    entropies = {}  # entropy for each query's subranking.
    for qid in query_ids:
        full_ranking = total_rankings[qid]
        # Ensure there are at least c documents.
        if len(full_ranking) < c:
            raise ValueError(f"Ranking for query {qid} has fewer than {c} documents.")
        subranking = full_ranking[:c]
        subrankings[qid] = subranking
        entropies[qid] = compute_entropy_for_subranking(subranking, score_lookup, qid)
    
    # Compute initial global statistics.
    S = sum(entropies[qid] for qid in query_ids)
    Q = sum(entropies[qid] ** 2 for qid in query_ids)
    global_mean = S / num_queries
    global_variance = Q / num_queries - (global_mean ** 2)
    
    iteration = 0
    improvement = float('inf')

    while iteration < max_iter and improvement > tol:
        prev_global_variance = global_variance
        improvement_made = False
        
        # Iterate over each query.
        for qid in query_ids:
            current_subranking = subrankings[qid]
            current_entropy = entropies[qid]
            best_candidate = current_subranking
            best_candidate_entropy = current_entropy
            best_global_variance = global_variance

            full_ranking = total_rankings[qid]
            # Available candidates are those documents not already in the current subranking.
            available_docs = [doc for doc in full_ranking if doc not in current_subranking]

            # Greedy swap: for each position in the current subranking,
            # try swapping with each candidate from available_docs.
            for pos in range(c):
                for candidate_doc in available_docs:
                    candidate_subranking = current_subranking.copy()
                    candidate_subranking[pos] = candidate_doc
                    candidate_entropy = compute_entropy_for_subranking(candidate_subranking, score_lookup, qid)
                    
                    # Update global statistics marginally: replace current_entropy with candidate_entropy.
                    new_S = S - current_entropy + candidate_entropy
                    new_Q = Q - (current_entropy ** 2) + (candidate_entropy ** 2)
                    new_mean = new_S / num_queries
                    new_variance = new_Q / num_queries - (new_mean ** 2)
                    
                    if new_variance > best_global_variance:
                        best_global_variance = new_variance
                        best_candidate = candidate_subranking
                        best_candidate_entropy = candidate_entropy

            # If a better candidate was found for this query, update the subranking and global stats.
            if best_global_variance > global_variance:
                # Update global statistics S and Q.
                S = S - current_entropy + best_candidate_entropy
                Q = Q - (current_entropy ** 2) + (best_candidate_entropy ** 2)
                global_mean = S / num_queries
                global_variance = Q / num_queries - (global_mean ** 2)
                # Update current query's subranking and entropy.
                subrankings[qid] = best_candidate
                entropies[qid] = best_candidate_entropy
                improvement_made = True

        improvement = global_variance - prev_global_variance
        iteration += 1
        
        # Stop early if no improvement was made.
        if not improvement_made:
            break

    return subrankings