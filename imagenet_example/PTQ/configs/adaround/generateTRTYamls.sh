#!/bin/bash

generate_yaml() {
    local sq_enabled=$1
    local alpha=$2
    local suffix=$3
    local filename="FYP_r18_8_8_trt_${suffix}.yaml"

    cat > "${filename}" << YAML
extra_prepare_dict:
    extra_qconfig_dict:
        w_observer: MSEObserver
        a_observer: EMAMSEObserver
        w_fakequantize: AdaRoundFakeQuantize
        a_fakequantize: FixedFakeQuantize
        w_qscheme:
            bit: 8
            symmetry: True
            per_channel: True
            pot_scale: False
            p: 2.4
        a_qscheme:
            bit: 8
            symmetry: True
            per_channel: False
            pot_scale: False
            p: 2.4
quantize:
    smoothquant:
        enabled: ${sq_enabled}
        alpha: ${alpha}
    backend: 'Tensorrt'
    quantize_type: advanced_ptq
    cali_batchsize: 16
    reconstruction:
        pattern: layer
        scale_lr: 4.0e-5
        warm_up: 0.2
        weight: 0.01
        max_count: 20000
        b_range: [20,2]
        keep_gpu: True
        round_mode: learned_hard_sigmoid
        prob: 1.0
    deploy:
        output_path: /home/users/ntu/sooq0001
        model_name: 'res18_8_8_${suffix}'
        deploy_to_qlinear: True
model:
    type: resnet18
    kwargs:
        num_classes: 1000
    path: /home/users/ntu/sooq0001/mydir/resnet18_imagenet.pth.tar
data:
    path: /home/users/ntu/sooq0001/scratch/ImageNet-ILSVRC2012/
    batch_size: 64
    num_workers: 4
    pin_memory: True
    input_size: 224
    test_resize: 256
process:
    seed: 1005
YAML
    echo "Created ${filename}"
}

generate_yaml false 0   NOSQ
generate_yaml true  0.5 SQ05
generate_yaml true  0.7 SQ07
