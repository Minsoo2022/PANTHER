#!/bin/bash

gpuid=$1
config=$2

### Dataset Information
#declare -a dataroots=(
#	'path/to/tcga_luad'
#)

dataroots='/data/e130bf/wsi_feat/TCGA-ALL-x256-x20-features-from_pan_cancer-from_dino-from_sagemaker-vitb16_dino-epoch=6-bf16/pt_files'

task='LUAD_survival'
target_col='dss_survival_days'
split_names='train,test_tcgaluad'

split_dir='survival/TCGA_LUAD_overall_survival_k=0'
#bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names $dataroots