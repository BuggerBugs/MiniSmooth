#!/bin/bash

# ── CHANGE THIS ONLY ────────────────────────────────────────────────
MODEL="r50"        # change to r50 when needed
# ─────────────────────────────────────────────────────────────────────

BASE_DIR="/home/users/ntu/sooq0001/mydir/MQBench/imagenet_example/PTQ"
RUNNER="${BASE_DIR}/ptq/runExperiments.py"
CONFIGS="${BASE_DIR}/configs"

submit() {
    local run_name=$1
    local config=$2

    qsub <<EOF
#!/bin/bash
#PBS -N ${run_name}
#PBS -l select=1:ngpus=1:mem=440G
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o out_run.txt
#PBS -q normal
module load miniforge3
conda activate mqbench
module load cuda/12.2.1
cd ${BASE_DIR}/ptq
python3 ${RUNNER} --run_name ${run_name} --config ${config}
EOF
    echo "Submitted: ${run_name}"
}

# ── TRT (adaround only) ─────────────────────────────────────────────
# for suffix in NOSQ SQ05 SQ07; do
#     submit "${MODEL}_8_8_trt_${suffix}" \
#            "${CONFIGS}/adaround/FYP_${MODEL}_8_8_trt_${suffix}.yaml"
# done

# ── adaround, brecq, qdrop ──────────────────────────────────────────
#for method in adaround brecq qdrop; do
#    for bits in 4_4; do
#        for suffix in NOSQ SQ05 SQ07; do
#            submit "${method}_${MODEL}_${bits}_${suffix}" \
#                   "${CONFIGS}/${method}/FYP_${MODEL}_${bits}_${suffix}.yaml"
#        done
#    done
#done

# ── min_max ─────────────────────────────────────────────────────────
#for bits in 5_5 6_6 8_8 8_4; do
#    for suffix in NOSQ SQ05 SQ07; do
#        submit "min_max_${MODEL}_${bits}_${suffix}" \
#               "${CONFIGS}/min_max/FYP_${MODEL}_${bits}_${suffix}.yaml"
#    done
#done

# ── mse ─────────────────────────────────────────────────────────
for bits in 8_4; do
    for suffix in NOSQ SQ05 SQ07; do
        submit "mse_${MODEL}_${bits}_${suffix}" \
               "${CONFIGS}/mse/FYP_${MODEL}_${bits}_${suffix}.yaml"
    done
done


