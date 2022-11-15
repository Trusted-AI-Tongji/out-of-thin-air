from openpose_1 import Openpose  # Out of thin air Attack
# from openpose_1_1 import Openpose  # Any area
# from openpose_2 import Openpose  # COCO data
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train openpose")
    parser.add_argument("-r", "--resume", help="whether resume from the latest saved model", action="store_true")
    parser.add_argument("-save", "--from_save_folder", help="whether resume from the save path", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    openpose = Openpose()
    
    if args.resume:
        openpose.resume_training_load(from_save_folder=args.from_save_folder)
    openpose.train(resume=True)  # resume=True加载之前训练的模型
