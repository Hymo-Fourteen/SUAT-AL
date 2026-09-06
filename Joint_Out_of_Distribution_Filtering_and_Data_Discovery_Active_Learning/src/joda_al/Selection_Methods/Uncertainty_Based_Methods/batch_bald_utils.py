import math
from dataclasses import dataclass
from typing import Union, List

import torch
from toma import toma
from tqdm import tqdm

"""
FROM
For an introduction & more information, see http://batchbald.ml/. The paper can be found at http://arxiv.org/abs/1906.08158.

The original implementation used in the paper is available at https://github.com/BlackHC/BatchBALD.
"""

from joda_al.Selection_Methods.Uncertainty_Based_Methods.joint_entropy import DynamicJointEntropy


@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]


def compute_entropy_2D(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C, W, H = log_probs_N_K_C.shape

    # entropies_N = torch.empty((N,W,H), dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    # @toma.execute.chunked(log_probs_N_K_C, 1024)
    # def compute(log_probs_n_K_C, start: int, end: int):
    #     mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
    #     nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)
    #
    #     entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
    #     pbar.update(end - start)


    mean_log_probs_n_C = torch.logsumexp(log_probs_N_K_C, dim=1) - math.log(K)
    nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)
    entropies_N=-torch.sum(nats_n_C, dim=1)
    pbar.close()

    return entropies_N


def compute_conditional_entropy_2D(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C, W, H = log_probs_N_K_C.shape

    entropies_N = torch.empty((N,W,H), dtype=torch.double)

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    # @toma.execute.chunked(log_probs_N_K_C, 1024)
    # def compute(log_probs_n_K_C, start: int, end: int):
    #     nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)
    #
    #     entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
    #     pbar.update(end - start)
    #
    # pbar.close()
    nats_n_K_C = log_probs_N_K_C * torch.exp(log_probs_N_K_C)
    entropies_N=-torch.sum(nats_n_K_C, dim=(1, 2)) / K

    return entropies_N


def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)

    pbar.close()

    return entropies_N

def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

    return entropies_N

def get_bald_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, dtype=None, device=None) -> Union[CandidateBatch,torch.tensor]:
    # This actually calculates the mutual infomation
    if len(log_probs_N_K_C.shape) == 5:
        entropy_f = compute_entropy_2D
        con_entropy_f = compute_conditional_entropy_2D
    else:  # len(log_probs_N_K_C.shape)== 3:
        entropy_f = compute_entropy
        con_entropy_f = compute_conditional_entropy
    N = log_probs_N_K_C.shape[0]

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    scores_N = -con_entropy_f(log_probs_N_K_C)
    scores_N += entropy_f(log_probs_N_K_C)

    candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)

    return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist()), scores_N


def get_entropy_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, dtype=None, device=None) -> Union[CandidateBatch,torch.tensor]:
    if len(log_probs_N_K_C.shape)== 5:
        entropy_f=compute_entropy_2D
    else: # len(log_probs_N_K_C.shape)== 3:
        entropy_f=compute_entropy

    N=log_probs_N_K_C.shape[0]
    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    scores_N = entropy_f(log_probs_N_K_C)

    candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)

    return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist()), scores_N

def get_batchbald_batch(
    log_probs_N_K_C: torch.Tensor, batch_size: int, num_samples: int, dtype=None, device=None
) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    if batch_size == 0:
        return CandidateBatch(candidate_scores, candidate_indices)

    conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C)

    batch_joint_entropy = DynamicJointEntropy(
        num_samples, batch_size - 1, K, C, dtype=dtype, device=device
    )

    # We always keep these on the CPU.
    scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

    for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])

        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

        batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

        scores_N -= conditional_entropies_N + shared_conditinal_entropies
        scores_N[candidate_indices] = -float("inf")

        candidate_score, candidate_index = scores_N.max(dim=0)

        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())

    return CandidateBatch(candidate_scores, candidate_indices)