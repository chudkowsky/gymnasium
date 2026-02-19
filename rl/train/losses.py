"""PPO loss functions."""
import torch
import torch.nn.functional as F


def compute_policy_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    old_logprobs: torch.Tensor,
    masks: torch.Tensor,
    clip_ratio: float = 0.2,
):
    """
    Compute clipped PPO policy loss.
    
    Args:
        logits: (N, 4100) model output logits
        actions: (N,) selected actions
        advantages: (N,) advantages from GAE
        old_logprobs: (N,) log probs under old policy
        masks: (N, 4100) legal action masks
        clip_ratio: PPO clip epsilon
        
    Returns:
        policy_loss: scalar loss
        approx_kl: approximate KL divergence
    """
    # Masked softmax
    masked_logits = logits.clone()
    masked_logits[masks == 0] = -float('inf')
    new_probs = torch.softmax(masked_logits, dim=-1)
    new_logprobs = torch.log(new_probs[torch.arange(len(actions)), actions] + 1e-10)
    
    # PPO clipped objective
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
    # Approximate KL
    approx_kl = (old_logprobs - new_logprobs).mean()
    
    return policy_loss, approx_kl


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_ratio: float = 0.2,
):
    """
    Compute value loss (MSE with optional clipping).
    
    Args:
        values: (N,) predicted values
        returns: (N,) target returns
        clip_ratio: optional clipping ratio
        
    Returns:
        value_loss: scalar loss
    """
    value_loss = F.mse_loss(values, returns)
    return value_loss


def compute_kl_penalty(
    current_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence from current policy to frozen reference policy.

    KL(π_current || π_ref) penalises the policy for drifting too far from the
    pretrained distribution, preventing catastrophic forgetting.

    Args:
        current_logits: (N, 4100) logits from current (trainable) policy
        ref_logits:     (N, 4100) logits from frozen reference policy
        masks:          (N, 4100) legal action masks

    Returns:
        kl: scalar mean KL divergence
    """
    masked_current = current_logits.clone()
    masked_ref = ref_logits.clone()
    masked_current[masks == 0] = -float('inf')
    masked_ref[masks == 0] = -float('inf')

    current_probs = torch.softmax(masked_current, dim=-1)
    ref_probs = torch.softmax(masked_ref, dim=-1)

    # KL(current || ref) = Σ current * log(current / ref)
    kl = (current_probs * (torch.log(current_probs + 1e-10) - torch.log(ref_probs + 1e-10))).sum(dim=-1)
    return kl.mean()


def compute_entropy_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
):
    """
    Compute entropy regularization loss.
    
    Args:
        logits: (N, 4100) model output logits
        masks: (N, 4100) legal action masks
        
    Returns:
        entropy_loss: negative entropy (to maximize entropy)
    """
    masked_logits = logits.clone()
    masked_logits[masks == 0] = -float('inf')
    probs = torch.softmax(masked_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
    return -entropy  # Negate because we want to maximize
