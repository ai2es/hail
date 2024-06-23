# Gridded Severe Hail Nowcasting Using 3D U-Nets, Lightning Observations, and the Warn-on-Forecast System

This is one of two repositories for the code and U-Nets associated with this study. The other repository is located at https://github.com/ai2es/auto-patcher. The U-Net is only located in the hail repository.

## Directories Most Relevant to the Reader

1. All primary code is located in each “src” directory. The code for the U-Net training is located in models.py in the hail repository.

2. The trained U-Net is located in the “model_data” directory in the hail repository. It has two copies, one is a TensorfFlow checkpoint, and the other is a h5 file.

## Other Directories

- Configs and OU supercomputing scheduling scripts are located in the “configs” and “slurm” directories respectively.

- “REU_CONTENT” contains content from the undergrad project that proceeded this one and was created by a different author. It has been kept for legacy reasons but it does not contain much overlap with the work done for this study at this stage.
