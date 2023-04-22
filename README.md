# Refactored implementation of paper "MotionCNN: A Strong Baseline for Motion Prediction in Autonomous Driving"
![MotionCNN Neural Network Scheme](img/header.png)
This repository contains updated code for our team's solution of Waymo Motion Prediction Challenge 2021 where we have achieved 3rd place.

* [📄 Paper [pdf]](https://arxiv.org/abs/2206.02163)
* [🎤 Presentation [pdf]](img/waymo_motion_prediction_2021_3rd_place_solution_presentation.pdf)
* [🎤 Announcement [youtube]](https://youtu.be/eOL_rCK59ZI?t=6485)
* [🚗 Waymo Motion Prediction Challenge website](https://arxiv.org/abs/2206.02163)
* [👩‍🏫 CVPR2021 Workshop on Autonomous Driving](http://cvpr2021.wad.vision/)
* [❗UPDATE: Related repo with 3rd place solution code for Waymo Motion Prediction Challenge 2022](https://github.com/stepankonev/waymo-motion-prediction-challenge-2022-multipath-plus-plus)

If you find this repo helpful feel free to share and ⭐️ it

## Related repos
* [Older version of this repo](https://github.com/kbrodt/waymo-motion-prediction-2021)
* [Kaggle Lyft motion prediciton 3rd place solution](https://gdude.de/blog/2021-02-05/Kaggle-Lyft-solution)

 ## Team behind this solution:
Listed as in the paper
* Stepan Konev
    [[LinkedIn]](https://www.linkedin.com/in/stepan-konev/)
    [[Twitter]](https://twitter.com/artsiom_s)
    [[GitHub]](https://github.com/kbrodt)
* Kirill Brodt
    [[GitHub]](https://github.com/kbrodt)
* Artsiom Sanakoyeu
    [[Homepage]](https://gdude.de)
    [[Twitter]](https://twitter.com/artsiom_s)
    [[Telegram Channel]](https://t.me/gradientdude)
    [[LinkedIn]](https://www.linkedin.com/in/sanakoev)


## Dataset

Download
[datasets](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_0_0)
`uncompressed/tf_example/{training,validation,testing}`

## Training and prerendering
In order to train the model first you need to prepare the dataset in a convenient format
```bash
python prerender.py \
    --data-path path/to/original/split \
    --output-path path/to/preprocessed/split \
    --config path/to/config.yaml \
    --n-jobs 16 \
    --n-shards 8 \
    --shard-id 0 \
```
Rendering the `training` split without sharding might be very resource demanding, so we recommend to use sharding (the number of shards depends on your computer's configuration)

Once the dataset is preprocessed, you can run the training script
```bash
python train.py \
    --train-data-path path/to/preprocessed/training/split \
    --val-data-path path/to/preprocessed/validation/split \
    --checkpoints-path path/to/save/checkpoints \
    --config path/to/config.yaml \
    [--multi-gpu]
```

## TODO:
Recently a Waymo Open Motion Dataset support was added to [trajdata](https://github.com/NVlabs/trajdata) repo, that provides a unified way to work with different motion datasets. We aim to refactor this code to consume `trajdata` format

## Citation
```
@misc{konev2022motioncnn,
      title={MotionCNN: A Strong Baseline for Motion Prediction in Autonomous Driving}, 
      author={Stepan Konev and Kirill Brodt and Artsiom Sanakoyeu},
      year={2022},
      eprint={2206.02163},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```