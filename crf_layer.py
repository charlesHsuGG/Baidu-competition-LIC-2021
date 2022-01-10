# -*- coding: utf-8 -*-
"""This is an CRF Modules based on pytorch implement."""
__author__ = "Charles_Hsu"

from typing import Optional

import torch
import torch.nn as nn


class CRFLayer(nn.Module):
    """Conditional random field Tool for torch LIb.

    Attributes:
        num_tags (int): Number of tags.
        average_batch (bool): Whether loss is average, default is True
    """

    def __init__(self, num_tags: int, average_batch: bool = True) -> None:
        """Init crf module.

        Arg:
            num_tags (int): Emission score tensor of size (batch_size, seq_length, num_tags).
            average_batch (bool, optional): average strategy. if average_batch is True, it will normalize log-likehood. Default is True.
        """
        super().__init__()
        self.num_tags = num_tags
        self.average_batch = average_batch

        init_start_transitions = torch.empty(num_tags)
        init_end_transitions = torch.empty(num_tags)
        init_transitions = torch.empty(num_tags, num_tags)

        self.start_transitions = nn.Parameter(init_start_transitions)
        self.end_transitions = nn.Parameter(init_end_transitions)
        self.transitions = nn.Parameter(init_transitions)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        """Print module to string."""
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute crf normalizer."""
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()

        if emissions.dim() != 3 and mask.dim() != 2:
            raise AssertionError("Incompatible Input Dimension. emissions.dim() != 3 and mask.dim() != 2")
        if emissions.shape[:2] != mask.shape:
            raise AssertionError("Incompatible Input Dimension. emissions.shape[:2] != mask.shape")
        if emissions.size(2) != self.num_tags:
            raise AssertionError("Incompatible Input Dimension. emissions.size(2) != self.num_tags")
        if not mask[0].all():
            raise AssertionError("Incompatible Input Dimension. not mask[0].all()")

        seq_length = emissions.size(0)

        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self.end_transitions

        return torch.logsumexp(score, dim=1)

    def viterbi_decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Viterbi decode to sequence.

        Arg:
            emissions (torch.Tensor): Emission score tensor of size (batch_size, seq_length, num_tags).
            mask (torch.Tensor): Mask tensor of size (batch_size, seq_length). Default to None.

        Returns:
            torch.Tensor: The Tensor tag sequence.
        """
        emissions = emissions.transpose(0, 1)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)
        else:
            mask = mask.transpose(0, 1)
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        if emissions.dim() != 3 and mask.dim() != 2:
            raise AssertionError("Incompatible Input Dimension. emissions.dim() != 3 and mask.dim() != 2")
        if emissions.shape[:2] != mask.shape:
            raise AssertionError("Incompatible Input Dimension. emissions.shape[:2] != mask.shape")
        if emissions.size(2) != self.num_tags:
            raise AssertionError("Incompatible Input Dimension. emissions.size(2) != self.num_tags")
        if not mask[0].all():
            raise AssertionError("Incompatible Input Dimension. not mask[0].all()")

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every
        # batch,
        # value at column j stores the score of the best tag sequence so far
        # that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this
        # is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best
        # tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags)
            # where
            # for each sample, entry at row i and column j stores the score of
            # the best
            # tag sequence so far that ends with transitioning from tag i to
            # tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices.tolist())

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = (mask.long().sum(dim=0) - 1).tolist()
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep;
            # this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag]

            # We trace back where the best last tag comes from, append that to
            # our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag)

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return torch.tensor(best_tags_list, dtype=torch.int32)

    def expected_log_likelihood(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        unary_marginals: torch.Tensor,
        pairwise_marginals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the expected log likelihood.

        Computes the expected log likelihood of CRFs defined by batch of emissions with respect to a reference distribution over the random variables.
        Arg:
            emissions (torch.Tensor): Emission score tensor of size (batch_size, seq_length, num_tags).
            mask (torch.Tensor): Mask tensor of size (batch_size, seq_length).
            unary_marginals (torch.Tensor): unary_marginals is marginal distributions over individual labels.
            pairwise_marginals (torch.Tensor, optional): pairwise_marginals a m x k x k tensor representing pairwise marginals over the ith and (i+1)th labels. Default to None.

        Returns:
            torch.Tensor: The Tensor tag sequence.
        """
        batch_size, seq_len, num_tags = emissions.size()

        partition = self._compute_normalizer(emissions, mask)

        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()

        unary_marginals = unary_marginals.transpose(0, 1)
        unary_marginals = unary_marginals.contiguous()
        if pairwise_marginals is not None:
            pairwise_marginals = pairwise_marginals.transpose(0, 1)
            pairwise_marginals = pairwise_marginals.contiguous()
        else:
            pairwise_marginals = torch.zeros(
                (seq_len - 1, batch_size, num_tags, num_tags), device=emissions.device.type
            )

            for i in range(seq_len - 1):
                for j in range(batch_size):
                    temp1 = unary_marginals[i, j]
                    temp2 = unary_marginals[i + 1, j]
                    temp = torch.ger(temp1, temp2)
                    pairwise_marginals[i, j, :, :] = temp

        temp = self.start_transitions.unsqueeze(0)
        temp = temp * unary_marginals[0]
        score = temp.sum(dim=1)

        for i in range(seq_len - 1):
            temp = emissions[i] * unary_marginals[i]
            temp = temp.sum(dim=1)
            score += temp * mask[i]

            temp = self.transitions.unsqueeze(0)
            temp = temp * pairwise_marginals[i]
            temp = temp.sum(dim=2).sum(dim=1)
            score += temp * mask[i + 1]

        index0 = mask.sum(dim=0).long() - 1
        index1 = torch.arange(0, batch_size, dtype=torch.long)
        last_marginals = unary_marginals[index0, index1, :]

        temp = self.end_transitions.unsqueeze(0)
        temp = temp * last_marginals
        temp = temp.sum(dim=1)
        score += temp

        last_scores = emissions[-1] * unary_marginals[-1]
        last_scores = last_scores.sum(dim=1)
        score += last_scores * mask[-1]

        return torch.sum(score - partition)
