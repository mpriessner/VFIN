import argparse
import os
from shutil import rmtree, move
import random

parser = argparse.ArgumentParser()
parser.add_argument('--ffmpeg_dir', type=str, default='', help='path to ffmpeg.exe')
parser.add_argument('--videos_folder', type=str, required=True, help='path to the folder containing videos')
parser.add_argument('--dataset_folder', type=str, required=True, help='path to the output dataset folder')
parser.add_argument('--continue_process', type=str, default='False', choices=['False', 'True'],
                    help='path to the output dataset folder')
parser.add_argument('--img_width', type=int, default=360, help='output image width')
parser.add_argument('--img_height', type=int, default=640, help='output image height')
parser.add_argument('--train_test_val_split', type=tuple, default=(70, 20, 10),
                    help='train test split for custom dataset')

args = parser.parse_args()


def listdir(path):
    files = os.listdir(path)
    for file in ['.DS_Store']:
        if file in files:
            files.remove(file)
    files.sort()
    return files


if args.continue_process == 'False' and os.path.exists(args.dataset_folder):
    if input('dataset_folder exists, delete? [y, n]: ').lower() == 'y':
        rmtree(args.dataset_folder)
    else:
        exit(1)


extractPath = os.path.join(args.dataset_folder, 'extracted')
trainPath = os.path.join(args.dataset_folder, 'train')
testPath = os.path.join(args.dataset_folder, 'test')
validationPath = os.path.join(args.dataset_folder, 'validation')

if args.continue_process == 'False':
    os.makedirs(args.dataset_folder)
    for folder in ['train', 'test', 'validation']:
        os.mkdir(f'{args.dataset_folder}/{folder}')
os.mkdir(f'{args.dataset_folder}/extracted')

videos = listdir(args.videos_folder)
video_frames = {}
for i, video in enumerate(videos):
    video_extraction_path = os.path.join(extractPath, video.split('.')[0])
    os.mkdir(video_extraction_path)
    os.system(f"{os.path.join(args.ffmpeg_dir, 'ffmpeg')} -loglevel error "
              f"-i '{os.path.join(args.videos_folder, video)}' -vsync 0 "
              f"-q:v 2 '{video_extraction_path}/%09d.jpg'")
    video_frames[video] = listdir(video_extraction_path)
    print(f'\rProcessed {i+1}/{len(videos)}: {video}', end='', flush=True)
print()

total_section_count = sum([len(i)//12 for i in video_frames.values()])
val_count = int(total_section_count * (args.train_test_val_split[2] / 100))
test_count = int(total_section_count * (args.train_test_val_split[1] / 100))
train_count = total_section_count - val_count - test_count

total_section = range(total_section_count)
train_set = list(total_section)
val_set = random.sample(train_set, val_count)
for tmp in val_set:
    train_set.remove(tmp)
test_set = random.sample(train_set, test_count)
for tmp in test_set:
    train_set.remove(tmp)

video_frames = list(video_frames.values())
if args.continue_process == 'True':
    val_test_train_count = [len(listdir(f'{args.dataset_folder}/{folder}')) for folder in ['validation', 'test', 'train']]
else:
    val_test_train_count = [0, 0, 0]

for section_index, section in enumerate(total_section):
    if len(video_frames[0]) < 12:
        video_frames.pop(0)
        videos.pop(0)
    frames = video_frames[0][:12]
    if random.randint(0, 1):
        frames = frames[::-1]

    # dest = 'None'
    if section in val_set:
        dest = f'validation/{val_test_train_count[0]}'
        val_test_train_count[0] += 1
    if section in test_set:
        dest = f'test/{val_test_train_count[1]}'
        val_test_train_count[1] += 1
    if section in train_set:
        dest = f'train/{val_test_train_count[2]}'
        val_test_train_count[2] += 1

    os.mkdir(f'{args.dataset_folder}/{dest}')
    for i, frame in enumerate(frames):
        move(f"{args.dataset_folder}/extracted/{videos[0].split('.')[0]}/{frame}",
             f'{args.dataset_folder}/{dest}/{i}.jpg')
        video_frames[0].remove(frame)
    print(f'\r{section_index + 1}/{total_section_count}', end='', flush=True)
print()
rmtree(extractPath)
