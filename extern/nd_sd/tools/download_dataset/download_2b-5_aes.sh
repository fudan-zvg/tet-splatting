#!/bin/bash

img2dataset --url_list ./laion2b-dataset-5-aes --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder laion-2ben-5_aes --processes_count 100 --thread_count 64 --image_size 512\
            --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
            --min_image_size 512 \
             --save_additional_columns '["similarity","hash","punsafe","pwatermark","AESTHETIC_SCORE"]' --enable_wandb False
