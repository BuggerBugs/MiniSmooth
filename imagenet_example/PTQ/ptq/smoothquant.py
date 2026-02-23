import torch
import torch.nn as nn

def get_module_by_name(model, name):
    """Get module from dotted name path"""
    parts = name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def find_quantizer_for_conv(model, conv_name):
    """
    Find the fake quantizer that feeds a given conv by traversing the torch.fx graph.
    """
    graph = model.graph
    
    # Find the conv node in the graph
    conv_node = None
    for node in graph.nodes:
        if node.op == 'call_module' and node.target == conv_name:
            conv_node = node
            break
    
    if not conv_node:
        return None
    
    # Get the input to this conv
    if not conv_node.args:
        return None
    
    conv_input = conv_node.args[0]
    
    # Check if the input is a fake quantizer node
    if hasattr(conv_input, 'op') and conv_input.op == 'call_module':
        if 'fake_quant' in conv_input.target.lower():
            return conv_input.target
    
    return None


def apply_smoothquant_to_prepared_model(model, calib_loader, alpha=0.5, device='cuda'):
    """
    Apply SmoothQuant by modifying weights and storing scales.
    
    FIXED:
    - Correct scale direction: multiply weights by s, divide inputs by s
    - Use hooks during calibration (GraphModule compatible)
    - Proper module access with dotted paths
    """
    from mqbench.utils.state import disable_all
    
    model.eval()
    model = model.to(device)
    
    print("\n" + "="*80)
    print("APPLYING SMOOTHQUANT (FIXED VERSION)")
    print("="*80)
    
    # Step 1: Find all Conv modules to smooth
    conv_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.intrinsic.qat.ConvBnReLU2d, nn.intrinsic.qat.ConvBn2d,
                              nn.qat.Conv2d, nn.Conv2d)):
            if name == 'conv1':
                print(f"  Skip (first layer): {name}")
            elif 'downsample' in name:
                print(f"  Skip (downsample): {name}")
            elif '.conv1' in name:  
                print(f"  Skip (residual block conv1): {name}")
            else:
                conv_modules[name] = module
                print(f"  Found: {name} ({type(module).__name__})")
    
    print(f"\n  Total: {len(conv_modules)} modules to smooth")
    
    if len(conv_modules) == 0:
        print("\n  ⚠️  WARNING: No Conv modules found!")
        print("="*80 + "\n")
        return model
    
    # Step 2: Collect input statistics (with quantization disabled)
    print("\n" + "="*80)
    print("COLLECTING INPUT STATISTICS")
    print("="*80)
    disable_all(model)
    
    act_stats = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                inp_tensor = inp[0]
                if isinstance(inp_tensor, torch.Tensor):
                    act_max = inp_tensor.abs().amax(dim=(0, 2, 3))
                    if name in act_stats:
                        act_stats[name] = torch.maximum(act_stats[name], act_max.cpu())
                    else:
                        act_stats[name] = act_max.cpu()
        return hook
    
    for name, module in conv_modules.items():
        hooks.append(module.register_forward_hook(make_hook(name)))
    
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            x = batch if isinstance(batch, torch.Tensor) else batch[0]
            model(x.to(device))
            if (i + 1) % 10 == 0:
                print(f"  Processed batch {i+1}")
    
    for h in hooks:
        h.remove()

        
    print(f"\n  Collected stats for {len(act_stats)} modules")
    
    # ====================================================================
    # DUMP STATS TO FILE FOR VISUALIZATION
    # ====================================================================
    import json
    dump = {}
    for name in conv_modules.keys():
        if name not in act_stats:
            continue
        act_max = act_stats[name].clamp(min=1e-5)
        weight_max = conv_modules[name].weight.abs().amax(dim=(0, 2, 3)).clamp(min=1e-5).cpu()
        s = act_max.pow(alpha) / weight_max.pow(1 - alpha)
        dump[name] = {
            'act_before':    act_max.tolist(),
            'act_after':     (act_max / s).tolist(),
            'weight_before': weight_max.tolist(),
            'weight_after':  (weight_max * s).tolist(),
            'scale':         s.tolist(),
        }
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stats_path = os.environ.get('SQ_STATS_PATH', f'smoothquant_stats_{timestamp}.json')
    with open(stats_path, 'w') as f:
        json.dump(dump, f)
    print("Saved smoothquant_stats!! Remember to move it to correct ipynb directory for processing")
    
    
    # Step 3: Compute smoothing factors and modify weights
    print("\n" + "="*80)
    print("COMPUTING SMOOTHING FACTORS AND MODIFYING WEIGHTS")
    print("="*80)
    
    scale_factors = {}
    
    for name, module in conv_modules.items():
        if name not in act_stats:
            print(f"\n  ❌ Skip {name}: No stats collected")
            continue
        
        print(f"\n  Processing: {name}")
        
        act_max = act_stats[name].to(device).clamp(min=1e-5)
        
        # Get weight
        if hasattr(module, 'weight'):
            weight = module.weight
        else:
            print(f"    ❌ Skip: No weight attribute")
            continue
        
        # Compute weight max per input channel
        weight_max = weight.abs().amax(dim=(0, 2, 3)).clamp(min=1e-5)
        
        # Check dimensions
        if act_max.shape[0] != weight_max.shape[0]:
            print(f"    ❌ Skip: dimension mismatch act={act_max.shape} weight={weight_max.shape}")
            continue
        
        # FIXED: Correct formula - s = (act_max / weight_max)^alpha
        # This migrates quantization difficulty from activations to weights
        s = (act_max.pow(alpha) / weight_max.pow(1 - alpha)).detach()
        
        print(f"    act_max: mean={act_max.mean():.4f}, range=[{act_max.min():.4f}, {act_max.max():.4f}]")
        print(f"    weight_max: mean={weight_max.mean():.4f}, range=[{weight_max.min():.4f}, {weight_max.max():.4f}]")
        print(f"    smoothing s: mean={s.mean():.4f}, range=[{s.min():.4f}, {s.max():.4f}]")
        
        if s.max() > 10 or s.min() < 0.1:
            print(f"    ⚠️  WARNING: Extreme smoothing factors detected!")
        
        # FIXED: MULTIPLY weights by s (not divide!)
        # Formula: Y = (X / s) @ (W * s)
        with torch.no_grad():
            weight.mul_(s.view(1, -1, 1, 1))
        
        print(f"    ✓ Modified weights: multiplied by s")
        
        # Store inverse scale (will divide inputs by s during calibration)
        scale_factors[name] = 1.0 / s
    
    # Step 4: Store scales on quantizers for hook-based application
    print("\n" + "="*80)
    print("STORING SCALES ON QUANTIZERS")
    print("="*80)
    
    applied_count = 0
    failed_count = 0
    
    for conv_name in scale_factors.keys():
        inv_scale = scale_factors[conv_name]
        
        # Find the quantizer that feeds this conv
        quant_name = find_quantizer_for_conv(model, conv_name)
        
        if quant_name is None:
            print(f"\n  ⚠️  Could not find quantizer for {conv_name}")
            failed_count += 1
            continue
        
        print(f"\n  {conv_name} <- {quant_name}")
        
        try:
            # FIXED: Use proper path traversal
            quantizer = get_module_by_name(model, quant_name)
            
            # Store the inverse scale (we'll multiply inputs by this to divide by s)
            quantizer.register_buffer('_smoothquant_inv_scale', inv_scale.to(device))
            
            print(f"    ✓ Stored inv_scale on {quant_name}")
            print(f"      inv_scale: mean={inv_scale.mean():.4f}, range=[{inv_scale.min():.4f}, {inv_scale.max():.4f}]")
            applied_count += 1
            
        except Exception as e:
            print(f"    ❌ Failed to apply scale: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue
    
    print("\n" + "="*80)
    print(f"✓ SMOOTHQUANT COMPLETE")
    print(f"  Modified weights: {len(scale_factors)} convs")
    print(f"  Applied scales: {applied_count} quantizers")
    if failed_count > 0:
        print(f"  ⚠️  Failed: {failed_count} quantizers")
    print("\n  NOTE: Use apply_smoothquant_scaling_hooks() during calibration")
    print("="*80 + "\n")
    
    return model


def apply_smoothquant_scaling_hooks(model):
    """
    Apply hooks to quantizers that have _smoothquant_inv_scale to scale inputs during calibration.
    Call this BEFORE calibration.
    
    Returns: list of hooks 
    """
    hooks = []
    
    print("\n" + "="*80)
    print("APPLYING SMOOTHQUANT SCALING HOOKS")
    print("="*80)
    
    for name, module in model.named_modules():
        if hasattr(module, '_smoothquant_inv_scale'):
            def make_hook(inv_scale):
                def hook(module, inp):
                    if isinstance(inp, tuple) and len(inp) > 0:
                        x = inp[0]
                        if isinstance(x, torch.Tensor):
                            # Multiply by inv_scale to divide by s
                            x_scaled = x * inv_scale.view(1, -1, 1, 1)
                            return (x_scaled,) + inp[1:]
                    return inp
                return hook
            
            hook = module.register_forward_pre_hook(make_hook(module._smoothquant_inv_scale))
            hooks.append(hook)
            print(f"  ✓ Applied scaling hook to: {name}")
    
    print(f"\n  Total hooks applied: {len(hooks)}")
    print("="*80 + "\n")
    return hooks


