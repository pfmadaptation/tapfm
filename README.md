# Code for Single GPU Task Adaptation of Pathology Foundation Models (TAPFM)
**TAPFM** enables efficient adaptation of pre-trained pathology foundation models for downstream clinical tasks on standard hardware. The framework addresses the challenge of fine-tuning large-scale vision transformers for whole slide image analysis while maintaining computational efficiency and training stability.

**Supported Models and Applications**
TAPFM integrates seamlessly with leading pathology foundation models including **UNI**, **GigaPath**, and **H-Optimus-0**. The framework demonstrates strong performance on clinically relevant mutation prediction tasks, supporting both binary and multi-label classification scenarios for actionable genetic alterations in cancer diagnosis.

**Usage**
Training
```console
python training_tapfm.py \
    --pfm gpath \
    --target_list PIK3CA_Binary FGFR3_Binary \
    --splits_csv_path /path/to/splits.csv \
    --tile_coords_csv_path /path/to/coordinates.csv \
    --slides_dir_path /path/to/slides \
    --outdir ./results \
    --nepochs 20 \
    --lr 1e-6
```
