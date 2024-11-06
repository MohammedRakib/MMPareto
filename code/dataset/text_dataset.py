import os
import pickle
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class M3AEClipDataset(Dataset):

    def __init__(self, args, mode='train'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode
        self.dataset = args.dataset
        
        # Define the directories for CLIP or non-CLIP usage
        if args.dataset == "Food101":
            self.data_root = '/data1/zhangxiaohui/food101/'
            self.visual_feature_path = os.path.join(self.data_root, "visual", '{}_imgs/'.format(mode))
            self.text_feature_path = os.path.join(self.data_root, "text_token_clip", '{}_token/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_food.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_food.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_food.txt"
        elif args.dataset == "MVSA":
            self.data_root = '/home/rifat/MMKD-Text/data/MVSA/'
            self.visual_feature_path = os.path.join(self.data_root, "visual", '{}_imgs/'.format(mode))
            self.text_feature_path = os.path.join(self.data_root, "text_token_clip", '{}_token/'.format(mode))
            self.stat_path = "/home/rifat/MMKD-Text/data/MVSA/stat_mvsa.txt"
            self.train_txt = "/home/rifat/MMKD-Text/data/MVSA/my_train_mvsa.txt"
            self.test_txt = "/home/rifat/MMKD-Text/data/MVSA/my_test_mvsa.txt"
        elif args.dataset == "CUB":
            self.data_root = '/data1/zhangxiaohui/CUB_200_2011/'
            self.visual_feature_path = os.path.join(self.data_root, "visual", '{}_imgs/'.format(mode))
            self.text_feature_path = os.path.join(self.data_root, "text_token_clip", '{}_token/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_cub.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_cub.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_cub.txt"

        with open(self.stat_path, "r") as f1:
            classes = f1.readlines()
        
        classes = [sclass.strip() for sclass in classes]

        if mode == 'train':
            csv_file = self.train_txt
        elif mode == 'test':
            csv_file = self.test_txt

        with open(csv_file, "r") as f2:
            csv_reader = f2.readlines()
            for single_line in csv_reader:
                item = single_line.strip().split(".jpg ")
                token_path = os.path.join(self.text_feature_path, item[0] + '_token.npy')
                if args.dataset == "MVSA" or args.dataset == "Food101" or args.dataset == "CUB":
                    visual_path = os.path.join(self.visual_feature_path, item[0] + ".jpg")    
                else:
                    visual_path = os.path.join(self.visual_feature_path, item[0])

                if os.path.exists(token_path) and os.path.exists(visual_path):
                    data.append(item[0])
                    data2class[item[0]] = item[1]
                else:
                    continue

        self.classes = sorted(classes)
        print(self.classes)
        self.data2class = data2class
        self.av_files = []
        for item in data:
            self.av_files.append(item)

        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))

        # CLIP requires 224x224 image size
        self.preprocess_train = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.preprocess_test = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.av_files)

    def get_image(self, filename, filename2=None, mix_lambda=1):
        img = Image.open(filename)
        if self.mode == "train":
            image_tensor = self.preprocess_train(img)
        elif self.mode == "test":
            image_tensor = self.preprocess_test(img)
        return image_tensor

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Text (if CLIP is used, no padding mask)
        token_path = os.path.join(self.text_feature_path, av_file + '_token.npy')
        tokenizer = np.load(token_path)
        tokenizer = torch.tensor(tokenizer)

        # Visual
        if self.dataset == "MVSA" or self.dataset == "Food101" or self.dataset == "CUB":
            image = self.get_image(os.path.join(self.visual_feature_path, av_file + ".jpg"))
        else:
            visual_path = os.path.join(self.visual_feature_path, av_file)
            allimages = os.listdir(visual_path)
            file_num = len(allimages)
            image = self.get_image(os.path.join(visual_path, allimages[int(file_num / 2)]))

        label = self.classes.index(self.data2class[av_file])

        return image, tokenizer, label
    


class URFunny_Dataloader(object):
    def __init__(self):
        self.process = eval("_process_1")

    def train_dataloader(self, dataset, batch_size, shuffle, num_workers, pin_memory):
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          collate_fn=self.process)
    
    def valid_dataloader(self, dataset, batch_size, shuffle, num_workers, pin_memory):
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          collate_fn=self.process)
    
    def test_dataloader(self, dataset, batch_size, shuffle, num_workers, pin_memory):
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          collate_fn=self.process)

def _process_1(inputs):
    # Lists to store processed batch elements
    tokenized_texts = []
    visions = []
    audios = []
    labels = []
    
    # Collate each component in the new order
    for sample in inputs:
        audios.append(sample[0])             # Audio features
        visions.append(sample[1])            # Vision features
        tokenized_texts.append(sample[2])    # Text tokens
        labels.append(sample[3])             # Labels

    # Pad sequences for tokenized texts and audio if necessary
    tokenized_texts = pad_sequence(tokenized_texts, batch_first=True).long()
    visions = pad_sequence(visions, batch_first=True)
    audios = pad_sequence(audios, batch_first=True)
    labels = torch.tensor(labels).long().view(-1)  # Ensure labels are 1D

    return audios, visions, tokenized_texts, labels


class Humor_Dataset(Dataset):
    def __init__(self, mode='train', aligned=True, z_norm=False):
        
        self.data_root = '/home/rifat/MMKD-Text/data/UR-Funny/data/urfunny_punchline.pkl'
        self.mode = mode
        self.aligned = aligned
        self.z_norm = z_norm

        # Load data
        with open(self.data_root, 'rb') as f:
            all_data = pickle.load(f)

        assert mode in ['train', 'valid', 'test']
        self.dataset = self.drop_entry(all_data[mode])

    def __len__(self):
        return self.dataset['vision'].shape[0]

    def drop_entry(self, dataset):
        """
        Drop entries where there's no text in the data.
        """
        drop = []
        for idx, k in enumerate(dataset['text']):
            if k.sum() == 0:
                drop.append(idx)
        for modality in list(dataset.keys()):
            dataset[modality] = np.delete(dataset[modality], drop, 0)
        return dataset

    def __getitem__(self, index):
        # Load vision (image features), audio, and text data
        vision = torch.tensor(self.dataset['vision'][index]).float()
        audio = torch.tensor(self.dataset['audio'][index]).float()
        text = torch.tensor(self.dataset['text'][index]).float()

        # Handle missing/infinite values in all modalities
        # vision = torch.nan_to_num(vision, nan=0.0, posinf=1.0, neginf=-1.0)
        # audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        # text = torch.nan_to_num(text, nan=0.0, posinf=1.0, neginf=-1.0)


        # Handle missing/infinite values in audio
        audio[audio == -np.inf] = 0.0

        # Alignment
        if self.aligned:
            start = text.nonzero(as_tuple=False)[0][0]
            vision, audio, text = vision[start:], audio[start:], text[start:]
        else:
            vision, audio, text = vision[vision.nonzero()[0][0]:], audio[audio.nonzero()[0][0]:], text[text.nonzero()[0][0]:]

        # Z-normalization
        if self.z_norm:
            vision = torch.nan_to_num((vision - vision.mean(0)) / vision.std(0))
            audio = torch.nan_to_num((audio - audio.mean(0)) / audio.std(0))
            text = torch.nan_to_num((text - text.mean(0)) / text.std(0))

        # Classification label (binary)
        label = 1 if self.dataset['labels'][index] >= 1 else 0

        # Prepare tokenizer (text token) as torch tensor, placeholder padding mask
        tokenizer = text  # Assume preprocessed text tokens

        # Return values compatible with MVSAClipDataset: tokenizer, padding mask, vision, label, and index
        return audio, vision, tokenizer,label