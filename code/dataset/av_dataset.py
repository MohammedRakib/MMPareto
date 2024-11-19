import csv
import os
import random
import librosa
import numpy as np
import torch
import torch.nn.functional
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

    


class AVDataset_CD(Dataset):
  def __init__(self, mode='train'):
    classes = []
    self.data = []
    data2class = {}

    self.mode=mode
    self.visual_path = '/data/users/public/cremad/cremad/visual/'
    self.audio_path = '/data/users/public/cremad/cremad/audio/'
    self.stat_path = '/data/users/public/cremad/cremad/stat.csv'
    self.train_txt = '/data/users/public/cremad/cremad/train.csv'
    self.test_txt = '/data/users/public/cremad/cremad/test.csv'
    if mode == 'train':
        csv_file = self.train_txt
    else:
        csv_file = self.test_txt

    
    with open(self.stat_path, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])
    
    with open(csv_file) as f:
      csv_reader = csv.reader(f)
      for item in csv_reader:
        if item[1] in classes and os.path.exists(self.audio_path + item[0] + '.pt') and os.path.exists(
                        self.visual_path + '/' + item[0]):
            self.data.append(item[0])
            data2class[item[0]] = item[1]

    print('data load over')
    print(len(self.data))
    
    self.classes = sorted(classes)

    self.data2class = data2class
    self._init_atransform()
    print('# of files = %d ' % len(self.data))
    print('# of classes = %d' % len(self.classes))

    #Audio
    self.class_num = len(self.classes)

  def _init_atransform(self):
    self.aid_transform = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
    return len(self.data)

  
  def __getitem__(self, idx):
    datum = self.data[idx]

    # Audio
    fbank = torch.load(self.audio_path + datum + '.pt').unsqueeze(0)

    # Visual
    if self.mode == 'train':
        transf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transf = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    folder_path = self.visual_path + datum
    file_num = len(os.listdir(folder_path))
    pick_num = 2
    seg = int(file_num/pick_num)
    image_arr = []

    for i in range(pick_num):
      if self.mode == 'train':
        index = random.randint(i*seg + 1, i*seg + seg)
      else:
        index = i*seg + int(seg/2)
      path = folder_path + '/frame_000' + str(index).zfill(2) + '.jpg'
      # print(path)
      image_arr.append(transf(Image.open(path).convert('RGB')).unsqueeze(0))

    images = torch.cat(image_arr)

    return fbank, images, self.classes.index(self.data2class[datum])



class CremadDataset(Dataset):
    def __init__(self, mode='train', 
                 train_path='/home/rakib/Multi-modal-Imbalance/data/CREMAD/train.csv',
                 test_path='/home/rakib/Multi-modal-Imbalance/data/CREMAD/test.csv',
                 visual_path='/home/rakib/Multimodal-Datasets/CREMA-D/Image-01-FPS',
                 audio_path='/home/rakib/Multimodal-Datasets/CREMA-D/AudioWAV'):
        
        self.mode = mode
        self.class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}
        self.visual_path = visual_path
        self.audio_path = audio_path

        # Use the appropriate CSV file depending on the mode (train or test)
        csv_file = train_path if mode == 'train' else test_path

        self.image = []
        self.audio = []
        self.label = []

        # Load data from CSV
        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_path, item[0] + '.wav')
                visual_path = os.path.join(self.visual_path, item[0])
                
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(self.class_dict[item[1]])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Load label
        label = self.label[idx]

        ### Audio Processing ###
        # Load and process audio with librosa
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        # Ensure we have 3 seconds of audio by tiling the sample if needed
        resamples = np.tile(samples, 3)[:22050 * 3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        
        # Compute the STFT and log-scale the spectrogram
        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        
        # Convert the spectrogram to a torch tensor and add a channel dimension
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)

        ### Visual Processing ###
        # Define the transformations (different for train and test)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224) if self.mode == 'train' else transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip() if self.mode == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Sample 3 frames from the image directory
        visual_path = self.image[idx]
        image_samples = sorted(os.listdir(visual_path))  # Get all image files
        pick_num = 2  # Fixed number of frames like in AVDataset
        seg = int(len(image_samples) / pick_num)  # Evenly spaced frame selection
        
        image_arr = []
        for i in range(pick_num):
            tmp_index = int(seg * i)
            img = Image.open(os.path.join(visual_path, image_samples[tmp_index])).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0)  # Add a batch dimension for concatenation
            image_arr.append(img)

        # Concatenate the 3 sampled frames along the batch dimension
        images = torch.cat(image_arr, dim=0)

        ### Return the data in the format required by AVDataset ###
        return spectrogram, images, label

      
      
class AVMNIST(Dataset):
    def __init__(self, data_root='/home/rakib/Multimodal-Datasets/AV-MNIST/avmnist', mode='train'):
        super(AVMNIST, self).__init__()
        image_data_path = os.path.join(data_root, 'image')
        audio_data_path = os.path.join(data_root, 'audio')
        
        if mode == 'train':
            self.image = np.load(os.path.join(image_data_path, 'train_data.npy'))
            self.audio = np.load(os.path.join(audio_data_path, 'train_data.npy'))
            self.label = np.load(os.path.join(data_root, 'train_labels.npy'))
            
        elif mode == 'test':
            self.image = np.load(os.path.join(image_data_path, 'test_data.npy'))
            self.audio = np.load(os.path.join(audio_data_path, 'test_data.npy'))
            self.label = np.load(os.path.join(data_root, 'test_labels.npy'))

        self.length = len(self.image)
        
    def __getitem__(self, idx):
        # Get image and audio for the index
        image = self.image[idx]
        audio = self.audio[idx]
        label = self.label[idx]
        
        # Normalize image and audio
        image = image / 255.0
        audio = audio / 255.0
        
        # Reshape image and audio
        image = image.reshape(28, 28)  # Reshape to 28x28 for MNIST
        image = np.expand_dims(image, 0)  # Add channel dimension: (1, 28, 28)
        audio = np.expand_dims(audio, 0)  # Add channel dimension: (1, 28, 28)
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        audio = torch.from_numpy(audio).float()
        label = torch.tensor(label, dtype=torch.long)
        
        # Return the same format as AVDataset: (spectrogram, image_n, label, idx)
        return audio, image, label
    
    def __len__(self):
        return self.length
      

class VGGSound(Dataset):

    def __init__(self, mode='train'):
        self.mode = mode
        train_video_data = []
        train_audio_data = []
        test_video_data = []
        test_audio_data = []
        train_label = []
        test_label = []
        train_class = []
        test_class = []

        with open('/home/rakib/Multi-modal-Imbalance/data/VGGSound/vggsound.csv') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip the header
            
            for item in csv_reader:
                youtube_id = item[0]
                timestamp = "{:06d}".format(int(item[1]))  # Zero-padding the timestamp
                train_test_split = item[3]

                video_dir = os.path.join('/home/rakib/Multimodal-Datasets/VGGSound/video/frames', train_test_split, 'Image-{:02d}-FPS'.format(1), f'{youtube_id}_{timestamp}')
                audio_dir = os.path.join('/home/rakib/Multimodal-Datasets/VGGSound/audio', train_test_split, f'{youtube_id}_{timestamp}.wav')

                if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir)) > 3:
                    if train_test_split == 'train':
                        train_video_data.append(video_dir)
                        train_audio_data.append(audio_dir)
                        if item[2] not in train_class: 
                            train_class.append(item[2])
                        train_label.append(item[2])
                    elif train_test_split == 'test':
                        test_video_data.append(video_dir)
                        test_audio_data.append(audio_dir)
                        if item[2] not in test_class: 
                            test_class.append(item[2])
                        test_label.append(item[2])

        self.classes = train_class
        class_dict = dict(zip(self.classes, range(len(self.classes))))

        if mode == 'train':
            self.video = train_video_data
            self.audio = train_audio_data
            self.label = [class_dict[label] for label in train_label]
        elif mode == 'test':
            self.video = test_video_data
            self.audio = test_audio_data
            self.label = [class_dict[label] for label in test_label]

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        # Audio processing (using librosa to compute the spectrogram)
        sample, rate = librosa.load(self.audio[idx], sr=16000, mono=True)
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(0, rate * 5)
        new_sample = sample[start_point:start_point + rate * 5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        # Image transformations based on mode
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Image processing
        image_samples = os.listdir(self.video[idx])
        image_samples = sorted(image_samples)
        pick_num = 3  # Fixed number of frames to match AVDataset's behavior
        seg = int(len(image_samples) / pick_num)
        image_arr = []

        for i in range(pick_num):
            tmp_index = int(seg * i)
            img = Image.open(os.path.join(self.video[idx], image_samples[tmp_index])).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(1).float()  # Add channel dimension for concatenation
            image_arr.append(img)
            if i == 0:
                image_n = img
            else:
                image_n = torch.cat((image_n, img), 1)  # Concatenate along the channel dimension

        # Label
        label = self.label[idx]

        return spectrogram, image_n, label