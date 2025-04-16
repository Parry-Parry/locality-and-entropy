import math
from rankers._util import load_json, save_json
from fire import Fire

def entropy_func(scores):
    """
    Compute the entropy of a list of scores after normalizing to obtain a probability distribution.
    If the sum of scores is zero, returns 0.
    """
    total = sum(scores)
    if total == 0:
        return 0.0
    norm_scores = [s / total for s in scores]
    return -sum(p * math.log(p) for p in norm_scores if p > 0)

def compute_entropy_for_subranking(subranking, score_lookup, query_id):
    """
    Given a subranking (list of doc_ids) for a query,
    look up the score for each doc via score_lookup and compute the entropy.
    """
    scores = [score_lookup[query_id].get(doc_id, 0) for doc_id in subranking]
    return entropy_func(scores)

def optimize_subranking_for_query(full_ranking, score_lookup, query_id, c, mode="max", max_iter=100, tol=1e-6):
    """
    For a given query, choose a subranking of size c that optimizes the entropy.

    Parameters:
      full_ranking : List[doc_id]
          The full ranking (list of document IDs) for this query.
      score_lookup : dict
          A dictionary mapping query_id to a dictionary {doc_id: score}.
      query_id : str
          The query identifier.
      c : int
          The desired subranking size.
      mode : str
          "max" to maximize entropy or "min" to minimize entropy.
      max_iter : int
          Maximum number of iterations.
      tol : float
          Tolerance for terminating the coordinate updates.
      
    Returns:
      (subranking, final_entropy): A tuple containing the selected subranking and its entropy.
    """
    if len(full_ranking) < c:
        raise ValueError(f"Full ranking for query {query_id} has fewer than {c} documents.")

    # Initialization: Use the first c documents.
    current_subranking = full_ranking[:c]
    current_entropy = compute_entropy_for_subranking(current_subranking, score_lookup, query_id)
    
    iteration = 0
    improvement = float('inf')
    
    # In optimization, 'improvement' is defined differently based on the mode.
    # For 'max', we are looking for a positive difference; for 'min' we are looking for a negative difference.
    while iteration < max_iter and abs(improvement) > tol:
        best_candidate = current_subranking
        best_candidate_entropy = current_entropy
        improved = False
        
        # Available docs not in the current subranking.
        available_docs = [doc for doc in full_ranking if doc not in current_subranking]
        
        # Try swapping each document in current subranking with one from available_docs.
        for pos in range(c):
            for candidate_doc in available_docs:
                candidate_subranking = current_subranking.copy()
                candidate_subranking[pos] = candidate_doc
                candidate_entropy = compute_entropy_for_subranking(candidate_subranking, score_lookup, query_id)
                
                # Check improvement according to the selected mode.
                if mode == "max" and candidate_entropy > best_candidate_entropy:
                    best_candidate = candidate_subranking
                    best_candidate_entropy = candidate_entropy
                    improved = True
                elif mode == "min" and candidate_entropy < best_candidate_entropy:
                    best_candidate = candidate_subranking
                    best_candidate_entropy = candidate_entropy
                    improved = True
        
        # Calculate the improvement difference.
        improvement = best_candidate_entropy - current_entropy if mode == "max" else current_entropy - best_candidate_entropy
        
        if improved:
            current_subranking = best_candidate
            current_entropy = best_candidate_entropy

        iteration += 1
        
        # If no improvement, break early.
        if not improved:
            break

    return current_subranking, current_entropy

def optimize_all_subrankings(total_rankings, score_lookup, c, mode="max", max_iter=100, tol=1e-6):
    """
    For each query, choose the subranking that optimizes (maximizes or minimizes) the entropy.

    Parameters:
      total_rankings : dict
          A dictionary {query_id: List[doc_id]} representing the full rankings.
      score_lookup : dict
          A dictionary {query_id: {doc_id: score}} for document scores.
      c : int
          Desired subranking size.
      mode : str
          "max" to maximize entropy, "min" to minimize entropy.
      max_iter : int
          Maximum number of iterations for the coordinate descent.
      tol : float
          Tolerance for improvement.
  
    Returns:
      subrankings : dict
          A dictionary {query_id: List[doc_id]} containing the selected subrankings.
    """
    subrankings = {}

    for query_id, full_ranking in total_rankings.items():
        subranking, final_entropy = optimize_subranking_for_query(
            full_ranking, score_lookup, query_id, c, mode, max_iter, tol)
        subrankings[query_id] = subranking

    return subrankings

def main(index_path: str,
         score_path: str,
         output_path: str,
         c: int,
         mode: str = "max",
         max_iter: int = 100,
         tol: float = 1e-6):
    """
    Main function to optimize subrankings based on scores.

    Parameters:
      triples_path : str
          Path to the input triples file.
      score_path : str
          Path to the score lookup file.
      output_path : str
          Path to save the optimized subrankings.
      c : int
          Desired subranking size.
      mode : str
          "max" to maximize entropy, "min" to minimize entropy.
      max_iter : int
          Maximum number of iterations for optimization.
      tol : float
          Tolerance for improvement.
    """
    # Load triples and scores.

    score_lookup = load_json(score_path)
    

    # Optimize subrankings.
    optimized_subrankings = optimize_all_subrankings(total_rankings, score_lookup, c, mode, max_iter, tol)

    # Save the optimized subrankings.
    save_json(optimized_subrankings, output_path)

    print(f"Optimized subrankings saved to {output_path}")


if __name__ == '__main__':
    Fire(main)
