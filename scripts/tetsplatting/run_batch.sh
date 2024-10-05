export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=./pretrained_models/huggingface

process_string() {
    local input_str="$1"
    if [[ ! "$input_str" =~ ^"a DSLR photo of "|^"a zoomed out DSLR photo of " ]]; then
        input_str="a DSLR photo of $input_str"
    fi
    echo "$input_str"
}

readarray -t prompts < <(sed 's/\n$//' $3)
for ((i=$1; i<$2; i++));
do
    result=$(echo "${prompts[$i]}" | tr ' ' '_')
    result=$(echo "$result" | tr -d '"')

    prompt="${prompts[$i]}"

    geo_out=tetsplatting/geo
    geo_refine_out=tetsplatting/geo-refine
    tex_out=tetsplatting/tex
    exp_root_dir=./outputs

    echo $i. "${prompts[$i]}"
    echo $result

    # step.1
    checkpoint_path=$exp_root_dir/$geo_out/$result/ckpts/epoch=0-step=1000.ckpt
    if [ -f "$checkpoint_path" ]; then
        echo "Skipping step1 as checkpoint exists: $checkpoint_path"
    else
        echo "Running step1"
        python3 launch.py --config configs/nd-mv-tetsplatting/geo.yaml \
                --train --gpu $4 system.prompt_processor.prompt="$prompt"  use_timestamp=False \
                name=$geo_out \
                data.elevation_range="[5, 30]" \
                data.fovy_range="[40, 45]" \
                data.camera_distance_range="[0.8, 1.0]" \
                exp_root_dir=$exp_root_dir ${@:5}
    fi

    # step.2
    checkpoint_path=$exp_root_dir/$geo_refine_out/$result/ckpts/epoch=0-step=2000.ckpt
    if [ -f "$checkpoint_path" ]; then
        echo "Skipping step2 as checkpoint exists: $checkpoint_path"
    else
        echo "Running step2"
        python3 launch.py --config configs/nd-mv-tetsplatting/geo-refine.yaml  \
                --train --gpu $4 system.prompt_processor.prompt="$prompt"  use_timestamp=False  \
                name=$geo_refine_out \
                system.geometry_convert_from=$exp_root_dir/$geo_out/$result/ckpts/last.ckpt \
                exp_root_dir=$exp_root_dir ${@:5}
    fi

    # step.3
    checkpoint_path=$exp_root_dir/$tex_out/$result/ckpts/last.ckpt
    prompt=$(process_string "${prompts[$i]}")
    if [ -f "$checkpoint_path" ]; then
        echo "Skipping step3 as checkpoint exists: $checkpoint_path"
    else
        echo "Running step3"
        python3 launch.py --config configs/nd-mv-tetsplatting/tex.yaml  \
                name=$tex_out \
                system.prompt_processor.prompt="$prompt"  use_timestamp=False  \
                system.geometry_convert_from=$exp_root_dir/$geo_refine_out/$result/ckpts/last.ckpt \
                --train --gpu $4\
                exp_root_dir=$exp_root_dir ${@:5}
    fi
    
    # step.4
    result=$(echo "${prompt}" | tr ' ' '_')
    result=$(echo "$result" | tr -d '"')
    obj_path=$exp_root_dir/$tex_out/$result/save/it2000-export_fix/model.obj
    if [ -f "$obj_path" ]; then
        echo "Skipping step4 as obj exists: $obj_path"
    else
        echo "Running step4"
        python launch.py --config $exp_root_dir/$tex_out/$result/configs/parsed.yaml --export --gpu $4 \
            resume=$exp_root_dir/$tex_out/$result/ckpts/last.ckpt system.exporter_type=mesh-exporter \
            system.exporter.context_type=cuda exp_root_dir=$exp_root_dir ${@:5}
    fi
done
#bash ./scripts/tetsplatting/run_batch.sh 0 1 ./prompts_dmtet.txt 7