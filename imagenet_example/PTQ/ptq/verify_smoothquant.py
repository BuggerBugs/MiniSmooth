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

    conv_node = None
    for node in graph.nodes:
        if node.op == 'call_module' and node.target == conv_name:
            conv_node = node
            break

    if not conv_node:
        return None

    if not conv_node.args:
        return None

    conv_input = conv_node.args[0]

    if hasattr(conv_input, 'op') and conv_input.op == 'call_module':
        if 'fake_quant' in conv_input.target.lower():
            return conv_input.target

    return None


# =============================================================================
# Layer classification helper — shared by all collection functions
# so "would_smooth" is always identified with the same rules
# =============================================================================

def _classify_conv_layers(model):
    """
    Returns (would_smooth: set, skipped: set) using the same rules as
    apply_smoothquant_to_prepared_model.
    """
    would_smooth = set()
    skipped      = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.intrinsic.qat.ConvBnReLU2d, nn.intrinsic.qat.ConvBn2d,
                                nn.qat.Conv2d, nn.Conv2d)):
            if name == 'conv1' or 'downsample' in name or '.conv1' in name:
                skipped.add(name)
            else:
                would_smooth.add(name)
    return would_smooth, skipped


def apply_smoothquant_to_prepared_model(model, calib_loader, alpha=0.5, device='cuda'):
    """
    Apply SmoothQuant by modifying weights and storing scales.

    - Correct scale direction: multiply weights by s, divide inputs by s
    - Use hooks during calibration (GraphModule compatible)
    - Proper module access with dotted paths
    - JSON stats include actual measured weight_after and act_after_actual
      for BOTH smoothed and unsmoothed conv layers, so the contrast is visible
    """
    from mqbench.utils.state import disable_all

    model.eval()
    model = model.to(device)

    print("\n" + "="*80)
    print("APPLYING SMOOTHQUANT (FIXED VERSION)")
    print("="*80)

    # =========================================================================
    # Step 1: Find all Conv modules — split into smoothed vs skipped
    # =========================================================================
    conv_modules         = {}   # will be smoothed
    conv_modules_skipped = {}   # tracked but NOT smoothed

    for name, module in model.named_modules():
        if isinstance(module, (nn.intrinsic.qat.ConvBnReLU2d, nn.intrinsic.qat.ConvBn2d,
                              nn.qat.Conv2d, nn.Conv2d)):
            if name == 'conv1':
                print(f"  Skip (first layer):        {name}")
                conv_modules_skipped[name] = module
            elif 'downsample' in name:
                print(f"  Skip (downsample):         {name}")
                conv_modules_skipped[name] = module
            elif '.conv1' in name:
                print(f"  Skip (residual block conv1): {name}")
                conv_modules_skipped[name] = module
            else:
                conv_modules[name] = module
                print(f"  Found (will smooth):       {name} ({type(module).__name__})")

    print(f"\n  Total to smooth:  {len(conv_modules)}")
    print(f"  Total skipped:    {len(conv_modules_skipped)}")

    if len(conv_modules) == 0:
        print("\n  ⚠️  WARNING: No Conv modules found to smooth!")
        print("="*80 + "\n")
        return model

    all_tracked = {**conv_modules, **conv_modules_skipped}

    # =========================================================================
    # Step 2: Collect input activation statistics for ALL tracked convs
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: COLLECTING INPUT ACTIVATION STATISTICS (all tracked convs)")
    print("="*80)
    disable_all(model)

    act_stats = {}

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

    hooks = []
    for name, module in all_tracked.items():
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            x = batch if isinstance(batch, torch.Tensor) else batch[0]
            model(x.to(device))
            if (i + 1) % 10 == 0:
                print(f"  Processed batch {i+1}")

    for h in hooks:
        h.remove()

    print(f"\n  Collected stats for {len(act_stats)} modules "
          f"({len(conv_modules)} smoothed, {len(conv_modules_skipped)} skipped)")

    # =========================================================================
    # Step 3: Compute smoothing factors and modify weights (smoothed only)
    # Stash weight_stats_before BEFORE modification for all tracked convs.
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: COMPUTING SMOOTHING FACTORS AND MODIFYING WEIGHTS")
    print("="*80)

    scale_factors       = {}
    weight_stats_before = {}

    # Stash weights for ALL tracked convs before any modification
    for name, module in all_tracked.items():
        if hasattr(module, 'weight'):
            weight_stats_before[name] = module.weight.abs().amax(dim=(0, 2, 3)).clamp(min=1e-5).detach().cpu()

    # Only modify smoothed convs
    for name, module in conv_modules.items():
        if name not in act_stats:
            print(f"\n  ❌ Skip {name}: No stats collected")
            continue

        print(f"\n  Processing: {name}")

        act_max = act_stats[name].to(device).clamp(min=1e-5)

        if not hasattr(module, 'weight'):
            print(f"    ❌ Skip: No weight attribute")
            continue

        weight     = module.weight
        weight_max = weight_stats_before[name].to(device)

        if act_max.shape[0] != weight_max.shape[0]:
            print(f"    ❌ Skip: dimension mismatch act={act_max.shape} weight={weight_max.shape}")
            continue

        s = (act_max.pow(alpha) / weight_max.pow(1 - alpha)).detach()

        print(f"    act_max:     mean={act_max.mean():.4f}, range=[{act_max.min():.4f}, {act_max.max():.4f}]")
        print(f"    weight_max:  mean={weight_max.mean():.4f}, range=[{weight_max.min():.4f}, {weight_max.max():.4f}]")
        print(f"    smoothing s: mean={s.mean():.4f}, range=[{s.min():.4f}, {s.max():.4f}]")

        if s.max() > 10 or s.min() < 0.1:
            print(f"    ⚠️  WARNING: Extreme smoothing factors detected!")

        with torch.no_grad():
            weight.mul_(s.view(1, -1, 1, 1))

        print(f"    ✓ Modified weights: multiplied by s")

        scale_factors[name] = 1.0 / s

    # =========================================================================
    # Step 4: Store inv_scale buffers on quantizers (smoothed only)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: STORING SCALES ON QUANTIZERS")
    print("="*80)

    applied_count = 0
    failed_count  = 0

    for conv_name in scale_factors.keys():
        inv_scale  = scale_factors[conv_name]
        quant_name = find_quantizer_for_conv(model, conv_name)

        if quant_name is None:
            print(f"\n  ⚠️  Could not find quantizer for {conv_name}")
            failed_count += 1
            continue

        print(f"\n  {conv_name} <- {quant_name}")

        try:
            quantizer = get_module_by_name(model, quant_name)
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
    print(f"  Applied scales:   {applied_count} quantizers")
    if failed_count > 0:
        print(f"  ⚠️  Failed: {failed_count} quantizers")
    print("\n  NOTE: Use apply_smoothquant_scaling_hooks() during calibration")
    print("="*80 + "\n")

    # =========================================================================
    # Step 5: Build and save JSON stats for ALL tracked convs
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: SAVING STATS JSON (smoothed + unsmoothed)")
    print("="*80)

    import json, os
    from datetime import datetime

    dump = {}

    for name, module in all_tracked.items():
        if name not in act_stats or name not in weight_stats_before:
            continue

        is_smoothed = name in conv_modules
        act_max     = act_stats[name].clamp(min=1e-5)
        w_before    = weight_stats_before[name].clamp(min=1e-5)
        w_after     = module.weight.abs().amax(dim=(0, 2, 3)).clamp(min=1e-5).detach().cpu()

        s = act_max.pow(alpha) / w_before.pow(1 - alpha)

        dump[name] = {
            'smoothed':            is_smoothed,
            'act_before':          act_max.tolist(),
            'act_after_theory':    (act_max / s).tolist(),
            'act_after_actual':    None,
            'weight_before':       w_before.tolist(),
            'weight_after_theory': (w_before * s).tolist(),
            'weight_after_actual': w_after.tolist(),
            'scale':               s.tolist(),
        }

    print("\n  Measuring actual post-smoothing activation stats (all tracked convs)...")

    temp_sq_hooks = []
    for conv_name in scale_factors.keys():
        quant_name = find_quantizer_for_conv(model, conv_name)
        if quant_name is None:
            continue
        try:
            quantizer = get_module_by_name(model, quant_name)
            if not hasattr(quantizer, '_smoothquant_inv_scale'):
                continue
            inv_scale = quantizer._smoothquant_inv_scale

            def make_sq_hook(s):
                def hook(module, inp):
                    if isinstance(inp, tuple) and len(inp) > 0:
                        x = inp[0]
                        if isinstance(x, torch.Tensor):
                            return (x * s.view(1, -1, 1, 1),) + inp[1:]
                    return inp
                return hook

            temp_sq_hooks.append(quantizer.register_forward_pre_hook(make_sq_hook(inv_scale)))
        except Exception:
            continue

    act_stats_after = {}

    def make_after_hook(name):
        def hook(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                inp_tensor = inp[0]
                if isinstance(inp_tensor, torch.Tensor):
                    act_max = inp_tensor.abs().amax(dim=(0, 2, 3))
                    if name in act_stats_after:
                        act_stats_after[name] = torch.maximum(act_stats_after[name], act_max.cpu())
                    else:
                        act_stats_after[name] = act_max.cpu()
        return hook

    after_hooks = []
    for name, module in all_tracked.items():
        after_hooks.append(module.register_forward_hook(make_after_hook(name)))

    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            x = batch if isinstance(batch, torch.Tensor) else batch[0]
            model(x.to(device))
            if (i + 1) % 10 == 0:
                print(f"  Processed batch {i+1}")

    for h in temp_sq_hooks + after_hooks:
        h.remove()

    print(f"  Collected actual post-smoothing stats for {len(act_stats_after)} modules")

    for name in dump.keys():
        if name in act_stats_after:
            dump[name]['act_after_actual'] = act_stats_after[name].clamp(min=1e-5).tolist()

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    stats_path = os.environ.get('SQ_STATS_PATH', f'smoothquant_stats_{timestamp}.json')
    with open(stats_path, 'w') as f:
        json.dump(dump, f)

    n_smoothed   = sum(1 for v in dump.values() if v['smoothed'])
    n_unsmoothed = sum(1 for v in dump.values() if not v['smoothed'])
    print(f"\n  Saved stats to: {stats_path}")
    print(f"  Smoothed layers:   {n_smoothed}")
    print(f"  Unsmoothed layers: {n_unsmoothed}")
    print("  Fields: smoothed, act_before, act_after_theory, act_after_actual,")
    print("          weight_before, weight_after_theory, weight_after_actual, scale")
    print("  Remember to move it to correct ipynb directory for processing")

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results


def _run_collection_pass(model, data_loader, all_conv_names, no_quantizer_layers, device):
    """
    Shared forward-pass logic used by both collect_post_quant_stats and
    collect_baseline_quant_stats.

    Registers hooks, runs the loader, returns:
        act_pre_quant_max, act_pre_quant_mean,
        act_post_quant_max, act_post_quant_mean,
        final_top1, final_top5, total_samples
    """
    act_pre_quant_max   = {}
    act_pre_quant_mean  = {}
    act_post_quant_max  = {}
    act_post_quant_mean = {}

    def _accumulate(store_max, store_mean, name, x):
        x_abs   = x.detach().abs()
        ch_max  = x_abs.amax(dim=(0, 2, 3)).cpu()
        ch_mean = x_abs.mean(dim=(0, 2, 3)).cpu()
        if name not in store_max:
            store_max[name]  = ch_max
            store_mean[name] = [ch_mean]
        else:
            store_max[name] = torch.maximum(store_max[name], ch_max)
            store_mean[name].append(ch_mean)

    def make_quant_hook(conv_name):
        def hook(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                x_pre = inp[0]
                if isinstance(x_pre, torch.Tensor) and x_pre.dim() == 4:
                    _accumulate(act_pre_quant_max, act_pre_quant_mean, conv_name, x_pre)
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                _accumulate(act_post_quant_max, act_post_quant_mean, conv_name, out)
        return hook

    def make_conv_pre_hook(conv_name):
        def hook(module, inp):
            if isinstance(inp, tuple) and len(inp) > 0:
                x = inp[0]
                if isinstance(x, torch.Tensor) and x.dim() == 4:
                    _accumulate(act_pre_quant_max, act_pre_quant_mean, conv_name, x)
        return hook

    hooks = []
    for conv_name in all_conv_names:
        quant_name = find_quantizer_for_conv(model, conv_name)
        if quant_name is not None:
            quantizer = get_module_by_name(model, quant_name)
            hooks.append(quantizer.register_forward_hook(make_quant_hook(conv_name)))
        else:
            conv_module = get_module_by_name(model, conv_name)
            hooks.append(conv_module.register_forward_pre_hook(make_conv_pre_hook(conv_name)))

    top1_correct = 0
    top5_correct = 0
    total        = 0
    track_acc    = True

    print(f"\n  Running forward pass over {len(data_loader)} batches...")
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if isinstance(batch, torch.Tensor):
                images    = batch
                track_acc = False
            else:
                images, target = batch[0], batch[1]
                target = target.to(device)

            images = images.to(device)
            output = model(images)

            if track_acc:
                acc1, acc5   = accuracy(output, target, topk=(1, 5))
                batch_size   = images.size(0)
                top1_correct += acc1[0].item() * batch_size / 100.0
                top5_correct += acc5[0].item() * batch_size / 100.0
                total        += batch_size

            if (i + 1) % 10 == 0:
                if track_acc and total > 0:
                    print(f"    Batch {i + 1:4d}  |  "
                          f"Top-1: {100. * top1_correct / total:.2f}%  "
                          f"Top-5: {100. * top5_correct / total:.2f}%")
                else:
                    print(f"    Batch {i + 1}")

    for h in hooks:
        h.remove()

    if track_acc and total > 0:
        final_top1 = 100. * top1_correct / total
        final_top5 = 100. * top5_correct / total
        print(f"\n  * Acc@1 {final_top1:.3f}%   Acc@5 {final_top5:.3f}%")
    else:
        final_top1 = final_top5 = None
        print("\n  ⚠️  No labels in data_loader — accuracy not computed")

    # Finalise mean accumulators
    act_pre_quant_mean  = {k: torch.stack(v).mean(dim=0) for k, v in act_pre_quant_mean.items()}
    act_post_quant_mean = {k: torch.stack(v).mean(dim=0) for k, v in act_post_quant_mean.items()}

    return (act_pre_quant_max, act_pre_quant_mean,
            act_post_quant_max, act_post_quant_mean,
            final_top1, final_top5, total)


def _assemble_and_save(
    model, results_prefix, file_env_var, file_prefix,
    all_conv_names, would_smooth_names, no_quantizer_layers,
    weight_stats,
    act_pre_quant_max, act_pre_quant_mean,
    act_post_quant_max, act_post_quant_mean,
    final_top1, final_top5, total,
):
    """Shared JSON assembly + save logic."""
    import json, os
    from datetime import datetime

    def _max_or_none(store, name):
        return store[name].tolist() if name in store else None

    def _mean_or_none(store, name):
        return store[name].tolist() if name in store else None

    results = {
        '_accuracy': {
            'top1':          final_top1,
            'top5':          final_top5,
            'total_samples': total,
        }
    }

    for name in all_conv_names:
        is_no_quant = name in no_quantizer_layers
        entry = {
            # "would_smooth" is used for both the SQ and baseline JSONs
            # so the notebook can align them by layer name + smoothing intent
            'would_smooth':        name in would_smooth_names,
            'act_pre_quant_max':   _max_or_none(act_pre_quant_max,  name),
            'act_pre_quant_mean':  _mean_or_none(act_pre_quant_mean, name),
            'act_post_quant_max':  _max_or_none(act_post_quant_max,  name) if not is_no_quant else None,
            'act_post_quant_mean': _mean_or_none(act_post_quant_mean, name) if not is_no_quant else None,
        }
        entry.update(weight_stats.get(name, {
            'weight_actual_max':  None,
            'weight_actual_mean': None,
        }))
        results[name] = entry

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    sq_path    = os.environ.get(file_env_var, f'{file_prefix}_{timestamp}.json')
    with open(sq_path, 'w') as f:
        json.dump(results, f, indent=2)

    n_ws  = sum(1 for k, v in results.items() if k != '_accuracy' and     v['would_smooth'])
    n_nws = sum(1 for k, v in results.items() if k != '_accuracy' and not v['would_smooth'])
    print(f"\n  Saved → {sq_path}")
    print(f"  Would-smooth: {n_ws}  |  Would-NOT-smooth: {n_nws}")
    print("  Fields per layer: would_smooth,")
    print("                    act_pre_quant_max/mean  (pre-fake-quantize),")
    print("                    act_post_quant_max/mean (post-fake-quantize, what conv sees),")
    print("                    weight_actual_max/mean")
    print("=" * 80 + "\n")

    return results


def collect_post_quant_stats(model, data_loader, device='cuda'):
    """
    Collect activation + weight stats after SmoothQuant + quantization.

    The SmoothQuant scaling pre-hooks must already be registered (via
    apply_smoothquant_scaling_hooks) before calling this.

    Saves to POST_QUANT_STATS_PATH env var or post_quant_stats_{timestamp}.json

    JSON schema per layer:
        would_smooth          bool   — True if this layer was (or would be) smoothed
        act_pre_quant_max     list   — per-channel max AFTER scaling, BEFORE fake-quantize
        act_pre_quant_mean    list   — per-channel mean (same point)
        act_post_quant_max    list   — per-channel max AFTER fake-quantize (what conv sees)
        act_post_quant_mean   list   — per-channel mean (same point)
        weight_actual_max     list   — per-output-channel weight max
        weight_actual_mean    list   — per-output-channel weight mean
    """
    import os

    model.eval()
    model = model.to(device)

    print("\n" + "=" * 80)
    print("COLLECTING POST-QUANT STATS  (SmoothQuant run)")
    print("=" * 80)

    would_smooth, skipped = _classify_conv_layers(model)
    all_conv_names        = would_smooth | skipped

    print(f"\n  Would-smooth: {len(would_smooth)}   Skipped: {len(skipped)}")

    # Identify layers with no quantizer in the graph (fallback hooking)
    no_quantizer_layers = [
        n for n in all_conv_names
        if find_quantizer_for_conv(model, n) is None
    ]
    if no_quantizer_layers:
        print(f"\n  ⚠️  No quantizer found for {len(no_quantizer_layers)} layers "
              f"(conv input hooked directly, act_post_quant will be None):")
        for n in no_quantizer_layers:
            print(f"      {n}")

    # Snapshot weights once
    weight_stats = {}
    for name, module in model.named_modules():
        if name not in all_conv_names or not hasattr(module, 'weight'):
            continue
        w_abs = module.weight.detach().abs()
        weight_stats[name] = {
            'weight_actual_max':  w_abs.amax(dim=(1, 2, 3)).cpu().tolist(),
            'weight_actual_mean': w_abs.mean(dim=(1, 2, 3)).cpu().tolist(),
        }

    (act_pre_max, act_pre_mean,
     act_post_max, act_post_mean,
     top1, top5, total) = _run_collection_pass(
        model, data_loader, all_conv_names, no_quantizer_layers, device)

    return _assemble_and_save(
        model,
        results_prefix    = 'sq',
        file_env_var      = 'POST_QUANT_STATS_PATH',
        file_prefix       = 'post_quant_stats',
        all_conv_names    = all_conv_names,
        would_smooth_names= would_smooth,
        no_quantizer_layers = no_quantizer_layers,
        weight_stats      = weight_stats,
        act_pre_quant_max = act_pre_max,
        act_pre_quant_mean= act_pre_mean,
        act_post_quant_max= act_post_max,
        act_post_quant_mean=act_post_mean,
        final_top1        = top1,
        final_top5        = top5,
        total             = total,
    )


def collect_baseline_quant_stats(model, data_loader, device='cuda'):
    """
    Collect the SAME stats as collect_post_quant_stats but for a run WITHOUT
    SmoothQuant — no scaling hooks, raw activations straight into the quantizer.

    Layers are still tagged with 'would_smooth' using the same classification
    rules so you can align them with the SmoothQuant JSON in the notebook.

    Because there is no scaling here:
        act_pre_quant  = raw activation entering the quantizer (no inv_scale applied)
        act_post_quant = fake-quantized version of that raw activation

    Saves to BASELINE_QUANT_STATS_PATH env var or baseline_quant_stats_{timestamp}.json
    """
    model.eval()
    model = model.to(device)

    print("\n" + "=" * 80)
    print("COLLECTING BASELINE QUANT STATS  (no SmoothQuant)")
    print("=" * 80)

    would_smooth, skipped = _classify_conv_layers(model)
    all_conv_names        = would_smooth | skipped

    print(f"\n  Would-smooth: {len(would_smooth)}   Skipped: {len(skipped)}")

    # Safety check — warn if any SmoothQuant scaling hooks are still active
    sq_hook_count = sum(
        1 for _, m in model.named_modules()
        if hasattr(m, '_smoothquant_inv_scale') and len(m._forward_pre_hooks) > 0
    )
    if sq_hook_count > 0:
        print(f"\n  ⚠️  WARNING: {sq_hook_count} SmoothQuant scaling hooks appear to still be active!")
        print("      Call h.remove() on all hooks returned by apply_smoothquant_scaling_hooks()")
        print("      before calling collect_baseline_quant_stats, otherwise results will be wrong.\n")

    no_quantizer_layers = [
        n for n in all_conv_names
        if find_quantizer_for_conv(model, n) is None
    ]
    if no_quantizer_layers:
        print(f"\n  ⚠️  No quantizer found for {len(no_quantizer_layers)} layers "
              f"(conv input hooked directly, act_post_quant will be None):")
        for n in no_quantizer_layers:
            print(f"      {n}")

    weight_stats = {}
    for name, module in model.named_modules():
        if name not in all_conv_names or not hasattr(module, 'weight'):
            continue
        w_abs = module.weight.detach().abs()
        weight_stats[name] = {
            'weight_actual_max':  w_abs.amax(dim=(1, 2, 3)).cpu().tolist(),
            'weight_actual_mean': w_abs.mean(dim=(1, 2, 3)).cpu().tolist(),
        }

    (act_pre_max, act_pre_mean,
     act_post_max, act_post_mean,
     top1, top5, total) = _run_collection_pass(
        model, data_loader, all_conv_names, no_quantizer_layers, device)

    return _assemble_and_save(
        model,
        results_prefix     = 'baseline',
        file_env_var       = 'BASELINE_QUANT_STATS_PATH',
        file_prefix        = 'baseline_quant_stats',
        all_conv_names     = all_conv_names,
        would_smooth_names = would_smooth,
        no_quantizer_layers= no_quantizer_layers,
        weight_stats       = weight_stats,
        act_pre_quant_max  = act_pre_max,
        act_pre_quant_mean = act_pre_mean,
        act_post_quant_max = act_post_max,
        act_post_quant_mean= act_post_mean,
        final_top1         = top1,
        final_top5         = top5,
        total              = total,
    )
