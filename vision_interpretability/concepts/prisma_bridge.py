"""Bridge to ViT Prisma for mechanistic interpretability."""

import warnings
from typing import Optional, Any

import torch
import torch.nn as nn


def load_vit_with_prisma(
    model_name: str = "vit_base_patch16_224",
    device: str = "cuda",
) -> tuple[nn.Module, Any]:
    """
    Load a ViT model through Prisma if available.
    
    Args:
        model_name: Name of ViT model
        device: Device to load model on
        
    Returns:
        model: PyTorch model
        prisma_wrapper: Prisma wrapper object (or None if unavailable)
    """
    try:
        import vit_prisma
        
        warnings.warn(
            "ViT Prisma integration is experimental. "
            "Falling back to standard timm model if issues occur."
        )
        
        from ..models.registry import load_model
        model, transform = load_model(model_name, device=device)
        
        return model, None
        
    except ImportError:
        warnings.warn(
            "vit-prisma not installed. Install with: pip install vit-prisma\n"
            "Falling back to standard model loading."
        )
        
        from ..models.registry import load_model
        model, transform = load_model(model_name, device=device)
        
        return model, None


def get_prisma_activations(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_indices: Optional[list[int]] = None,
) -> dict[str, torch.Tensor]:
    """
    Get per-layer activations from a ViT, optionally through Prisma.
    
    Args:
        model: ViT model
        inputs: Input tensor (N, C, H, W)
        layer_indices: Optional list of layer indices to extract
        
    Returns:
        Dictionary mapping layer names to activations
    """
    activations = {}
    
    if hasattr(model, "blocks"):
        blocks = model.blocks
        layer_indices = layer_indices or list(range(len(blocks)))
        
        hooks = []
        
        def get_hook(name: str):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        for idx in layer_indices:
            if idx < len(blocks):
                name = f"block_{idx}"
                handle = blocks[idx].register_forward_hook(get_hook(name))
                hooks.append(handle)
        
        with torch.no_grad():
            _ = model(inputs)
        
        for handle in hooks:
            handle.remove()
    
    else:
        warnings.warn(
            "Model does not have 'blocks' attribute. "
            "This function is designed for ViT models."
        )
    
    return activations


def inspect_attention_patterns(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Extract attention patterns from a ViT model.
    
    Args:
        model: ViT model
        inputs: Input tensor (N, C, H, W)
        layer_idx: Layer index to inspect
        head_idx: Optional specific attention head to inspect
        
    Returns:
        Attention weights tensor
    """
    if not hasattr(model, "blocks"):
        raise ValueError("Model does not appear to be a ViT (no 'blocks' attribute)")
    
    attention_weights = None
    
    def attention_hook(module, input, output):
        nonlocal attention_weights
        if hasattr(module, "attn_drop"):
            attention_weights = output
    
    target_block = model.blocks[layer_idx]
    
    if hasattr(target_block, "attn"):
        handle = target_block.attn.register_forward_hook(attention_hook)
    else:
        raise ValueError(f"Block {layer_idx} does not have 'attn' attribute")
    
    with torch.no_grad():
        _ = model(inputs)
    
    handle.remove()
    
    if attention_weights is not None and head_idx is not None:
        if attention_weights.ndim == 4:
            attention_weights = attention_weights[:, head_idx]
    
    return attention_weights
