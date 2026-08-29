## Introduction to options
If you need more effects,  use '--option your-parameters' to enter what you need.

### Base

|    Option    |        Description         |                 Default                 |
| :----------: | :------------------------: | :-------------------------------------: |
|  --device   |   device for all models: auto / cuda / mps / directml / cpu   |                  auto                  |
|  --model    |   restoration model id/alias/checkpoint: quality / dm-baseline / traditional / legacy.pth / lite:path.pth / portable:path.pth   |                  auto                   |
|  --manifest |   extra model manifest yaml(s), merged over the built-in registry   |                  none                  |
|  --gpu_id   |   (legacy) if -1, do not use gpu; prefer --device    |                    0                    |
| --media_path | your videos or images path |            ./imgs/ruoruo.jpg            |
| --start_time | start position of video, default is the beginning of video | '00:00:00' |
| --last_time | limit the duration of the video, default is the entire video | '00:00:00' |
|    --mode    |    program running mode(auto/clean/add/style)    |                 'auto'                  |
| --model_path |   pretrained model path    | ./pretrained_models/mosaic/add_face.pth |
| --result_dir |  output media will be saved here|                 ./result          |
| --temp_dir | Temporary files will go here | ./tmp |
|    --fps    |    read and output fps, if 0-> origin    |                 0                  |
| --no_preview | if specified,do not preview images when processing video. eg.(when run it on server) | Flase |

### AddMosaic

|    Option    |        Description         |                 Default                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --mosaic_mod | type of mosaic -> squa_avg/ squa_random/ squa_avg_circle_edge/ rect_avg/random |                    squa_avg                    |
| --mosaic_size | mosaic size,if 0 -> auto size |            0            |
|    --mask_extend    |    extend mosaic area    |         10  |
| --mask_threshold | threshold of recognize mosaic position 0~255 (auto-adapted when --no_auto_adapt not set) | 48 |

### CleanMosaic

|    Option    |        Description         |                 Default                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --traditional | if specified, use traditional image processing methods to clean mosaic |                                        |
| --tr_blur | ksize of blur when using traditional method, it will affect final quality |            10            |
|    --tr_down    |    downsample when using traditional method,it will affect final quality    |         10  |
| --restore_strength | blend restored detail into source region: 0=source, 1=full model | 1.0 |
| --max_restore_side | maximum source-content crop side before model-required padding; 0 selects a memory-safe device profile | 0 |
| --restore_clip_len | maximum frames per neural call; 0 selects a memory-safe device profile | 0 |
| --all_mosaic_area | if specified, detect all mosaic regions per frame (multi-mosaic); otherwise only largest | false |
| --min_mosaic_area | minimum connected component area (pixels) to keep; auto-lowered when --no_auto_adapt not set | 150 |
| --min_mosaic_size | minimum bounding-box half-size (pixels) to clean; auto-lowered when --no_auto_adapt not set | 40 |
| --no_auto_adapt | disable automatic threshold/size adaptation; use fixed --mask_threshold/--min_mosaic_* | false |
| --medfilt_num | medfilt window of mosaic movement in the video | 5 |

### Style Transfer

|    Option    |        Description         |                 Default                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --output_size | size of output media, if 0 -> origin |512|
