# egodex
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/test_egodex.py \
  --dataset_path /mnt/raid0/EgoDex/test/add_remove_lid \
  --episode 0
```

# xperience-10m
```bash
python scripts/test_xperience_10m.py \
  --dataset_path /mnt/raid0/xperience-10m/003dcaf0-edba-4787-ada0-187d2748f684 \
  --episode 2 \
  --max_frames 1000
```

# egoVerse
```bash
python scripts/test_egoverse.py \
  --dataset_path /mnt/raid0/egocentric_datasets/egoVerse/69b57e518cd7957ebe4794cd \
  --max_frames 1000
```

# npz转换为ply
```bash
python tools/visualize_demo_result.py outputs/egodex_add_remove_lid_0_droid.npz \
  --frame-stride 2 \
  --pixel-stride 8 \
  --save-ply outputs_egodex/egodex_add_remove_lid_0.ply \
  --no-window
```