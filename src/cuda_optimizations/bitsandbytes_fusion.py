"""
FirstSight: BitsAndBytes Fusion Optimizations
Fused operations and optimizations for QLoRA training with bitsandbytes
"""

import logging
import torch
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def enable_bitsandbytes_fusion(
    use_fused_8bit: bool = True,
    use_fused_adam: bool = True,
    use_fused_lion: bool = False,
    optimize_quantization: bool = True
) -> Dict[str, bool]:
    """
    Enable bitsandbytes fused operations for optimized QLoRA training.
    
    Args:
        use_fused_8bit: Enable fused 8-bit optimizers (AdamW, Adam)
        use_fused_adam: Enable fused Adam optimizer (requires bitsandbytes>=0.41.0)
        use_fused_lion: Enable fused Lion optimizer (experimental)
        optimize_quantization: Enable quantization kernel optimizations
    
    Returns:
        Dict with enabled optimizations status
    """
    enabled = {}
    
    try:
        import bitsandbytes as bnb
        
        # Set environment variables for bitsandbytes optimizations
        if optimize_quantization:
            # Enable optimized quantization kernels
            os.environ['BITSANDBYTES_NOWELCOME'] = '1'
            # Use faster quantization paths
            os.environ['BITSANDBYTES_FAST'] = '1'
            enabled['quantization_optimization'] = True
            logger.info("✓ BitsAndBytes quantization optimizations enabled")
        
        # Check for fused optimizer support
        if use_fused_8bit:
            try:
                # Test if fused optimizers are available
                if hasattr(bnb.optim, 'AdamW8bit'):
                    enabled['fused_8bit_adamw'] = True
                    logger.info("✓ Fused 8-bit AdamW optimizer available")
                if hasattr(bnb.optim, 'Adam8bit'):
                    enabled['fused_8bit_adam'] = True
                    logger.info("✓ Fused 8-bit Adam optimizer available")
            except Exception as e:
                logger.warning(f"Fused 8-bit optimizers not available: {e}")
                enabled['fused_8bit_adamw'] = False
                enabled['fused_8bit_adam'] = False
        
        if use_fused_adam:
            try:
                if hasattr(bnb.optim, 'PagedAdamW'):
                    enabled['fused_paged_adamw'] = True
                    logger.info("✓ Fused PagedAdamW optimizer available")
            except Exception as e:
                logger.debug(f"PagedAdamW not available: {e}")
                enabled['fused_paged_adamw'] = False
        
        if use_fused_lion:
            try:
                if hasattr(bnb.optim, 'Lion8bit'):
                    enabled['fused_8bit_lion'] = True
                    logger.info("✓ Fused 8-bit Lion optimizer available")
            except Exception as e:
                logger.debug(f"Fused Lion optimizer not available: {e}")
                enabled['fused_8bit_lion'] = False
        
        return enabled
        
    except ImportError:
        logger.warning("bitsandbytes not available. Install with: pip install bitsandbytes")
        return {'error': 'bitsandbytes_not_installed'}


def get_fused_optimizer(
    optimizer_name: str,
    model_parameters,
    lr: float,
    **optimizer_kwargs
):
    """
    Get a fused bitsandbytes optimizer for QLoRA training.
    
    Args:
        optimizer_name: Name of optimizer ('adamw', 'adam', 'lion', 'paged_adamw')
        model_parameters: Model parameters to optimize
        lr: Learning rate
        **optimizer_kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer instance (fused if available, otherwise standard)
    """
    try:
        import bitsandbytes as bnb
        
        optimizer_name_lower = optimizer_name.lower()
        
        if optimizer_name_lower == 'adamw':
            if hasattr(bnb.optim, 'AdamW8bit'):
                logger.info("Using fused 8-bit AdamW optimizer")
                return bnb.optim.AdamW8bit(
                    model_parameters,
                    lr=lr,
                    **optimizer_kwargs
                )
            else:
                logger.info("Fused AdamW not available, using standard AdamW")
                from torch.optim import AdamW
                return AdamW(model_parameters, lr=lr, **optimizer_kwargs)
        
        elif optimizer_name_lower == 'adam':
            if hasattr(bnb.optim, 'Adam8bit'):
                logger.info("Using fused 8-bit Adam optimizer")
                return bnb.optim.Adam8bit(
                    model_parameters,
                    lr=lr,
                    **optimizer_kwargs
                )
            else:
                logger.info("Fused Adam not available, using standard Adam")
                from torch.optim import Adam
                return Adam(model_parameters, lr=lr, **optimizer_kwargs)
        
        elif optimizer_name_lower == 'paged_adamw':
            if hasattr(bnb.optim, 'PagedAdamW'):
                logger.info("Using fused PagedAdamW optimizer")
                return bnb.optim.PagedAdamW(
                    model_parameters,
                    lr=lr,
                    **optimizer_kwargs
                )
            else:
                # Fallback to AdamW8bit or standard AdamW
                return get_fused_optimizer('adamw', model_parameters, lr, **optimizer_kwargs)
        
        elif optimizer_name_lower == 'lion':
            if hasattr(bnb.optim, 'Lion8bit'):
                logger.info("Using fused 8-bit Lion optimizer")
                return bnb.optim.Lion8bit(
                    model_parameters,
                    lr=lr,
                    **optimizer_kwargs
                )
            else:
                logger.warning("Fused Lion not available, falling back to AdamW")
                return get_fused_optimizer('adamw', model_parameters, lr, **optimizer_kwargs)
        
        else:
            logger.warning(f"Unknown optimizer: {optimizer_name}, using standard AdamW")
            from torch.optim import AdamW
            return AdamW(model_parameters, lr=lr, **optimizer_kwargs)
            
    except ImportError:
        logger.warning("bitsandbytes not available, using standard optimizer")
        from torch.optim import AdamW
        return AdamW(model_parameters, lr=lr, **optimizer_kwargs)


def optimize_quantization_config(
    bnb_config: Any,
    use_double_quant: bool = True,
    use_fast_quantization: bool = True
) -> Any:
    """
    Optimize BitsAndBytesConfig for better performance.
    
    Args:
        bnb_config: Existing BitsAndBytesConfig or config dict
        use_double_quant: Enable double quantization for memory savings
        use_fast_quantization: Use faster quantization kernels
    
    Returns:
        Optimized BitsAndBytesConfig
    """
    try:
        from transformers import BitsAndBytesConfig
        
        # If it's already a config, return as-is (already optimized)
        if isinstance(bnb_config, BitsAndBytesConfig):
            return bnb_config
        
        # If it's a dict, convert to config
        if isinstance(bnb_config, dict):
            config_dict = bnb_config.copy()
            
            # Enable optimizations
            if use_double_quant and 'bnb_4bit_use_double_quant' not in config_dict:
                config_dict['bnb_4bit_use_double_quant'] = True
            
            return BitsAndBytesConfig(**config_dict)
        
        return bnb_config
        
    except Exception as e:
        logger.warning(f"Could not optimize quantization config: {e}")
        return bnb_config


def check_bitsandbytes_available() -> bool:
    """Check if bitsandbytes is available."""
    try:
        import bitsandbytes
        return True
    except ImportError:
        return False


def get_bitsandbytes_version() -> Optional[str]:
    """Get bitsandbytes version if available."""
    try:
        import bitsandbytes
        return bitsandbytes.__version__
    except ImportError:
        return None

