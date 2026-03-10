#!/bin/bash

# Output directory — change this to wherever your FYP yamls live
OUTDIR="./"

mkdir -p "$OUTDIR"

generate() {
    local wbit=$1
    local abit=$2
    local sq_enabled=$3
    local alpha=$4
    local suffix=$5

    local filename="FYP_r50_${wbit}_${abit}_${suffix}.yaml"

    cat > "$OUTDIR/$filename" <<EOF
extra_prepare_dict:
    extra_qconfig_dict:
        w_observer: MSEObserver
        a_observer: EMAMSEObserver
        w_fakequantize: AdaRoundFakeQuantize
        a_fakequantize: FixedFakeQuantize
        w_qscheme:
            bit: ${wbit}
            symmetry: False
            per_channel: True
            pot_scale: False
            p: 2.4
        a_qscheme:
            bit: ${abit}
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
        pattern: layer
        scale_lr: 4.0e-5
        warm_up: 0.2
        weight: 0.01
        max_count: 20000
        b_range: [20,2]
        keep_gpu: True
        round_mode: learned_hard_sigmoid
        prob: 1.0
model:
    type: resnet50
    kwargs:
        num_classes: 1000
    path: /home/users/ntu/sooq0001/mydir/resnet50_imagenet.pth.tar
data:
    path: /home/users/ntu/sooq0001/scratch/ImageNet-ILSVRC2012/
    batch_size: 64
    num_workers: 4
    pin_memory: True
    input_size: 224
    test_resize: 256
process:
    seed: 1005
EOF

    echo "Created: $OUTDIR/$filename"
}

for wbit_abit in "8_4"; do
    wbit="${wbit_abit%_*}"
    abit="${wbit_abit#*_}"
    generate $wbit $abit false 0   NOSQ
    generate $wbit $abit true  0.5 SQ05
    generate $wbit $abit true  0.7 SQ07
done

echo "Done. Generated files"
