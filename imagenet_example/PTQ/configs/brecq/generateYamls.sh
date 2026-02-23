#!/bin/bash

generate_yaml() {
    local wbits=$1
    local abits=$2
    local sq_enabled=$3
    local alpha=$4
    local suffix=$5
    local filename="FYP_r18_${wbits}_${abits}_${suffix}.yaml"

    cat > "${filename}" << YAML
extra_prepare_dict:
    extra_qconfig_dict:
        w_observer: MSEObserver
        a_observer: EMAMSEObserver
        w_fakequantize: AdaRoundFakeQuantize
        a_fakequantize: QDropFakeQuantize
        w_qscheme:
            bit: ${wbits}
            symmetry: False
            per_channel: True
            pot_scale: False
            p: 2.4
        a_qscheme:
            bit: ${abits}
            symmetry: False
            per_channel: False
            pot_scale: False
            p: 2.4
quantize:
    smoothquant:
        enabled: ${sq_enabled}
        alpha: ${alpha}
    quantize_type: advanced_ptq
    cali_batchsize: 16
    reconstruction:
        pattern: block
        scale_lr: 4.0e-5
        warm_up: 0.2
        weight: 0.01
        max_count: 20000
        b_range: [20,2]
        keep_gpu: True
        round_mode: learned_hard_sigmoid
        prob: 1.0
model:
    type: resnet18
    kwargs:
        num_classes: 1000
    path: /home/users/ntu/sooq0001/mydir/resnet18_imagenet.pth.tar
data:
    path: /home/users/ntu/sooq0001/scratch/ImageNet-ILSVRC2012/
    batch_size: 16
    num_workers: 4
    pin_memory: True
    input_size: 224
    test_resize: 256
process:
    seed: 1005
YAML
    echo "Created ${filename}"
}

# 2_4, 3_3, 4_4
generate_yaml 2 4 false 0   NOSQ
generate_yaml 2 4 true  0.5 SQ05
generate_yaml 2 4 true  0.7 SQ07

generate_yaml 3 3 false 0   NOSQ
generate_yaml 3 3 true  0.5 SQ05
generate_yaml 3 3 true  0.7 SQ07

generate_yaml 4 4 false 0   NOSQ
generate_yaml 4 4 true  0.5 SQ05
generate_yaml 4 4 true  0.7 SQ07
