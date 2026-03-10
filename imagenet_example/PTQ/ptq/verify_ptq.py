import numpy as np
import argparse
from data.imagenet import load_data
from models import load_model
from utils import parse_config, seed_all, evaluate
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.advanced_ptq import ptq_reconstruction
from mqbench.convert_deploy import convert_deploy

backend_dict = {
    'Academic': BackendType.Academic,
    'Tensorrt': BackendType.Tensorrt,
    'SNPE': BackendType.SNPE,
    'PPLW8A16': BackendType.PPLW8A16,
    'NNIE': BackendType.NNIE,
    'Vitis': BackendType.Vitis,
    'ONNX_QNN': BackendType.ONNX_QNN,
    'PPLCUDA': BackendType.PPLCUDA,
}


def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    for i, batch in enumerate(train_loader):
        cali_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    return cali_data


def get_quantize_model(model, config):
    backend_type = BackendType.Academic if not hasattr(
        config.quantize, 'backend') else backend_dict[config.quantize.backend]
    extra_prepare_dict = {} if not hasattr(
        config, 'extra_prepare_dict') else config.extra_prepare_dict
    return prepare_by_platform(
        model, backend_type, extra_prepare_dict)


def deploy(model, config):
    backend_type = BackendType.Academic if not hasattr(
        config.quantize, 'backend') else backend_dict[config.quantize.backend]
    output_path = './' if not hasattr(
        config.quantize, 'deploy') else config.quantize.deploy.output_path
    model_name = config.quantize.deploy.model_name
    deploy_to_qlinear = False if not hasattr(
        config.quantize.deploy, 'deploy_to_qlinear') else config.quantize.deploy.deploy_to_qlinear

    convert_deploy(model, backend_type, {
                   'input': [1, 3, 224, 224]}, output_path=output_path, model_name=model_name, deploy_to_qlinear=deploy_to_qlinear)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet Solver')
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()
    config = parse_config(args.config)
    # seed first
    seed_all(config.process.seed)
    # load_model
    model = load_model(config.model)
    if hasattr(config, 'quantize'):
        model = get_quantize_model(model, config)
        # load_data
        train_loader, val_loader = load_data(**config.data)

        # Generate calibration data
        if (config.quantize.quantize_type == 'advanced_ptq' or
                config.quantize.quantize_type == 'naive_ptq' or
                (hasattr(config.quantize, 'smoothquant') and config.quantize.smoothquant.enabled)):
            cali_data = load_calibrate_data(train_loader, cali_batchsize=config.quantize.cali_batchsize)
            
        # ══════════════════════════════════════════════════════════════════════
        # INJECT OUTLIERS (proof of concept) (cancel;led)
        # ══════════════════════════════════════════════════════════════════════
        #import torch
        #outlier_bias = torch.zeros(64).cuda()
        #outlier_bias[[0, 5, 10, 15]] = 1.2  # 4 channels with huge offset

        # Access via getattr for GraphModule
        #conv1_module = None
        #for name, module in model.named_modules():
        #    if name == 'layer1.0.conv1':
        #        conv1_module = module
        #        break

       # if conv1_module is not None:
       #     original_forward = conv1_module.forward
       #     def forward_with_outliers(x):
       #         out = original_forward(x)
       #         return out * outlier_bias.view(1, -1, 1, 1)
       #     conv1_module.forward = forward_with_outliers
       #     
       #     print("\n" + "="*80)
       #     print("INJECTED OUTLIERS: 4 channels at additional offset in layer1.0.conv1 output")
       #     print("="*80 + "\n")
       # else:
       #     print("⚠️  Could not find layer1.0.conv1 to inject outliers")

        # Apply SmoothQuant 
        if hasattr(config.quantize, 'smoothquant') and config.quantize.smoothquant.enabled:
            from mqbench.utils.state import disable_all
            disable_all(model)
            from verify_smoothquant import apply_smoothquant_to_prepared_model
            sq_alpha = config.quantize.smoothquant.alpha if hasattr(config.quantize.smoothquant, 'alpha') else 0.5
            model = apply_smoothquant_to_prepared_model(
                model,
                cali_data,
                alpha=sq_alpha,
                device='cuda'
            )
            from verify_smoothquant import apply_smoothquant_scaling_hooks
            hooks = apply_smoothquant_scaling_hooks(model)

    model.cuda()
    # evaluate
    if not hasattr(config, 'quantize'):
        evaluate(val_loader, model)
    elif config.quantize.quantize_type == 'advanced_ptq':
        print('begin calibration now!')
        from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
        model.eval()
        import torch
        with torch.no_grad():
           #################DEBUG SANITY CHECK CODE############ 
            # RIGHT BEFORE calibration starts:
        #    from smoothquant import get_module_by_name
        #    print("\n" + "="*80)
        #    print("VERIFYING CALIBRATION SEES SCALED ACTIVATIONS")
        #    print("="*80)

            # Always use this specific quantizer to test .trt resnet-18
        #    test_q_name = 'layer1_0_conv1_post_act_fake_quantizer'
        #    test_q = get_module_by_name(model, test_q_name)

            # Record min/max BEFORE calibration
        #    if hasattr(test_q, 'activation_post_process'):
        #        obs = test_q.activation_post_process
        #        print(f"Quantizer: {test_q_name}")
        #        print(f"  Observer min_val BEFORE calibration: {obs.min_val}")
        #        print(f"  Observer max_val BEFORE calibration: {obs.max_val}")
##############END OF SANITY CHECK CODE###################
            enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
            for batch in cali_data:
                model(batch.cuda())
            enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
            model(cali_data[0].cuda())
        print('begin advanced PTQ now!')
        if hasattr(config.quantize, 'reconstruction'):
            model = ptq_reconstruction(
                model, cali_data, config.quantize.reconstruction)
        enable_quantization(model)

        for name, module in model.named_modules():
                    if name == "":
                        continue  # skip top-level container
                    print(f"{name} -> {type(module).__name__}")

        # Verify hooks are registered
        #print("\n" + "="*80)
        #print("VERIFYING HOOKS ARE REGISTERED")
        #print("="*80)
        #hook_count = 0
        #for name, module in model.named_modules():
        #    if hasattr(module, '_forward_pre_hooks') and len(module._forward_pre_hooks) > 0:
        #        if hasattr(module, '_smoothquant_inv_scale'):
        #            print(f"  ✓ {name}: {len(module._forward_pre_hooks)} hook(s)")
        #            hook_count += 1
        #print(f"\nTotal modules with SmoothQuant hooks: {hook_count}")
        #print("="*80 + "\n")

        # ============================================================================
        # ADD TEST 1 HERE: Are hooks actually SCALING?
        # ============================================================================
        #print("\n" + "="*80)
        #print("TESTING IF HOOKS ACTUALLY SCALE INPUTS")
        #print("="*80)

        #test_module = None
        #test_name = None
        #for name, module in model.named_modules():
        #    if hasattr(module, '_smoothquant_inv_scale'):
        #        test_module = module
        #        test_name = name
        #        break

       # if test_module:
       #     print(f"Testing: {test_name}")
       #     print(f"inv_scale: {test_module._smoothquant_inv_scale[:5]}")
       #     
       #     C = test_module._smoothquant_inv_scale.shape[0]
       #     test_input = torch.randn(1, C, 32, 32).cuda()
       #     
       #     captured = {'input': None}
       #     
       #     def capture_hook(module, inp):
       #         captured['input'] = inp[0].clone()
       #     
       #     h = test_module.register_forward_pre_hook(capture_hook)
            
       #     with torch.no_grad():
       #         _ = test_module(test_input)
            
        #    h.remove()
            
        #    if captured['input'] is not None:
        #        original_means = test_input.mean(dim=(0,2,3))
        #        captured_means = captured['input'].mean(dim=(0,2,3))
        #        ratio = captured_means / original_means
        #        expected = test_module._smoothquant_inv_scale
                
        #        print(f"\nRatio (captured/original): {ratio[:5]}")
        #        print(f"Expected (inv_scale):      {expected[:5]}")
                
        #        if torch.allclose(ratio, expected, rtol=0.1):
        #            print("\n✓ Hook IS scaling inputs correctly!")
        #        else:
        #            print("\n❌ Hook NOT scaling! Inputs unchanged!")
        #            print(f"Max difference: {(ratio - expected).abs().max():.6f}")

        #print("="*80 + "\n")

        # ============================================================================
        # ADD TEST 2 HERE: Were weights modified?
        # ============================================================================
        #print("="*80)
        #print("CHECKING IF WEIGHTS WERE MODIFIED")
        #print("="*80)

        #for name, module in model.named_modules():
        #    if name == 'layer1.0.conv2':
        #        print(f"\n{name}:")
        #        print(f"  Weight mean: {module.weight.mean():.6f}")
        #        print(f"  Weight std: {module.weight.std():.6f}")
        #        print(f"  Weight abs mean: {module.weight.abs().mean():.6f}")
        #        print(f"  Weight range: [{module.weight.min():.6f}, {module.weight.max():.6f}]")
        #        print(f"\nIf SmoothQuant worked, these should be 2-5x larger than normal")
        #        break

        #print("="*80 + "\n")

        # ============================================================================
        # REMOVE HOOKS (comment this out to test WITH hooks)
        # ============================================================================
        #for name, module in model.named_modules():
        #    if hasattr(module, '_smoothquant_inv_scale'):
        #        module._forward_pre_hooks.clear()

        # ══════════════════════════════════════════════════════════════════════
        # REMOVE OUTLIERS (restore normal forward)
        # ══════════════════════════════════════════════════════════════════════
        #if conv1_module is not None:
        #    conv1_module.forward = original_forward
        #    print("\n" + "="*80)
        #    print("REMOVED OUTLIERS: restored normal forward")
        #    print("="*80 + "\n")
        ## ══════════════════════════════════════════════════════════════════════

        # ── Post-quantization stats collection ────────────────────────────────
        if hasattr(config.quantize, 'smoothquant') and config.quantize.smoothquant.enabled:
            # SmoothQuant run — scaling hooks are still active from apply_smoothquant_scaling_hooks()
            from verify_smoothquant import collect_post_quant_stats
            collect_post_quant_stats(model, val_loader, device='cuda')
        else:
            # Baseline (no SmoothQuant) run — collect identical stats for comparison
            # No scaling hooks exist so this is just raw quantized activations
            from verify_smoothquant import collect_baseline_quant_stats
            collect_baseline_quant_stats(model, val_loader, device='cuda')
        # ─────────────────────────────────────────────────────────────────────

        #train_loader, val_loader = load_data(**config.data)
        #evaluate(val_loader, model)
        if hasattr(config.quantize, 'deploy'):
            deploy(model, config)
    elif config.quantize.quantize_type == 'naive_ptq':
        print('begin calibration now!')
        from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
        # do activation and weight calibration seperately for quick MSE per-channel for weight one
        model.eval()
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
        for batch in cali_data:
            model(batch.cuda())
        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        model(cali_data[0].cuda())
        print('begin quantization now!')
        enable_quantization(model)

        # ============================================================================
        # REMOVE HOOKS (comment this out to test WITH hooks)
        # ============================================================================
        #for name, module in model.named_modules():
        #    if hasattr(module, '_smoothquant_inv_scale'):
        #     module._forward_pre_hooks.clear()

        # ── Post-quantization stats collection ────────────────────────────────
        if hasattr(config.quantize, 'smoothquant') and config.quantize.smoothquant.enabled:
            # SmoothQuant run — scaling hooks are still active from apply_smoothquant_scaling_hooks()
            from verify_smoothquant import collect_post_quant_stats
            collect_post_quant_stats(model, val_loader, device='cuda')
        else:
            # Baseline (no SmoothQuant) run — collect identical stats for comparison
            # No scaling hooks exist so this is just raw quantized activations
            from verify_smoothquant import collect_baseline_quant_stats
            collect_baseline_quant_stats(model, val_loader, device='cuda')
       
        # ─────────────────────────────────────────────────────────────────────
        #train_loader, val_loader = load_data(**config.data)
        #evaluate(val_loader, model)
        if hasattr(config.quantize, 'deploy'):
            deploy(model, config)
    else:
        print("The quantize_type must in 'naive_ptq' or 'advanced_ptq',")
        print("and 'advanced_ptq' need reconstruction configration.")


