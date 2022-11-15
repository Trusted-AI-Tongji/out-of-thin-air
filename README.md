# out-of-thin-air
## Requirements
    Python 3.7+
    pytorch-gpu 1.1+
    download weight file
    models/pretrained_vgg_base.pth  https://drive.google.com/file/d/1sSeOC9IvhrDCR9uOl0zin2-OrqRoJKD0/view?usp=share_link 
    models/posenet.pth  https://drive.google.com/file/d/1fXDuvL65oANhC-55UJuq5Ajj-vKKYVYj/view?usp=share_link
## Download COCO 2017 dataset（To save memory, you can operate the validation dataset image as the train dataset image）
    --data
      --coco
        --annotations
          -- person_keypoints_train2017.json
          -- person_keypoints_val2017.json
        --train2017
          -- 000000000139.jpg
          -- 000000000285.jpg
          -- ...
        --val2017
          -- 000000000139.jpg
          -- 000000000285.jpg
          -- ...
## (Recommended) Install with conda
    #1. Generate and save image masks
    python gen_ignore_mask.py
    #2. adversarial examples generation
    python train_adv.py
    
