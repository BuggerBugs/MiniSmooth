#!/bin/bash

generate_yaml() {
    local sq_enabled=$1
    local alpha=$2
    local suffix=$3
    local filename="FYP_r18_4_4_${suffix}.yaml"

    cat > "${filename}" << YAML
extra_prepare_dict:
    extra_qconfig_dict:
        w_observer: MinMaxObserver
        a_observer: EMAMinMaxObserver
        w_fakequantize: FixedFakeQuantize
        a_fakequantize: FixedFakeQuantize
        w_qscheme:
            bit: 4
            symmetry: False
            per_channel: True
            pot_scale: False
        a_qscheme:
            bit: 4
            symmetry: False
            per_channel: False
            pot_scale: False
quantize:
    smoothquant:
        enabled: ${sq_enabled}
        alpha: ${alpha}
    quantize_type: naive_ptq
    cali_batchsize: 16
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
