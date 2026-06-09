# pure distillation (use_task_rewards=False) ignores task reward, but verl's reward manager
# still runs over each rollout, so hand it a constant 0 to keep the empty ground_truth happy.
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return 0.0
