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

GPUS=$1
prompt=$2
SAVE_PATH=$3
exp_root_dir=$SAVE_PATH/tmp
SAVE_MODEL_PATH=$SAVE_PATH/result
mkdir -p  $exp_root_dir
mkdir -p  $SAVE_MODEL_PATH


echo "input prompt: $prompt"
result=$(echo "${prompt}" | tr ' ' '_')
result=$(echo "$result" | tr -d '"')
echo output_dir: $SAVE_PATH
echo GPUs: $GPUS

geo_out=nd-mv-tetsplatting/geo
geo_refine_out=nd-mv-tetsplatting/geo-refine
tex_out=nd-mv-tetsplatting/tex-fast


rm -rf $exp_root_dir/$geo_out/$result
rm -rf $exp_root_dir/$geo_refine_out/$result
rm -rf $exp_root_dir/$tex_out/a_DSLR_photo_of_$result
echo $exp_root_dir/$tex_out/$result

# step.1
python3 launch.py --config configs/nd-mv-tetsplatting/geo.yaml \
        --train --gpu $GPUS system.prompt_processor.prompt="$prompt"  use_timestamp=False \
        name=$geo_out \
        data.elevation_range="[5, 30]" \
        data.fovy_range="[40, 45]" \
        data.camera_distance_range="[0.8, 1.0]" \
        exp_root_dir=$exp_root_dir ${@:4}

# step.2
python3 launch.py --config configs/nd-mv-tetsplatting/geo-refine.yaml  \
        --train --gpu $GPUS  system.prompt_processor.prompt="$prompt"  use_timestamp=False  \
        name=$geo_refine_out \
        system.geometry_convert_from=$exp_root_dir/$geo_out/$result/ckpts/last.ckpt \
        exp_root_dir=$exp_root_dir ${@:4}

# step.3
ITER=2000
prompt=$(process_string "${prompt}") # trick proposed By Fantasia3D
python3 launch.py --config configs/nd-mv-tetsplatting/tex.yaml  \
        name=$tex_out \
        system.prompt_processor.prompt="$prompt"  use_timestamp=False  \
        system.geometry_convert_from=$exp_root_dir/$geo_refine_out/$result/ckpts/last.ckpt \
        --train --gpu $GPUS system.prompt_processor.prompt="$prompt"  use_timestamp=False \
        exp_root_dir=$exp_root_dir trainer.max_steps=$ITER ${@:4}


# extract mesh
result=$(echo "${prompt}" | tr ' ' '_')
result=$(echo "$result" | tr -d '"')
python launch.py --config $exp_root_dir/$tex_out/$result/configs/parsed.yaml --export --gpu $GPUS \
    resume=$exp_root_dir/$tex_out/$result/ckpts/last.ckpt system.exporter_type=mesh-exporter \
    system.exporter.context_type=cuda exp_root_dir=$exp_root_dir ${@:4}

# copy result
output_path=$exp_root_dir/$tex_out/$result/save/it$ITER-export/*
output_path=$(echo "$output_path" | tr -d '"')
cp -r $output_path $SAVE_MODEL_PATH