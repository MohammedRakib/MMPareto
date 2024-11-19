import numpy as np
import torch
import torch.nn as nn
import clip
import torch.nn.functional as F

from .backbone import resnet18


class Transformer(nn.Module):
    """
    Transformer-based encoder for each modality.
    """
    def __init__(self, n_features, dim, n_head, n_layers):
        super(Transformer, self).__init__()
        self.embed_dim = dim
        self.conv = nn.Conv1d(n_features, self.embed_dim, kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(self.embed_dim, nhead=n_head)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))  # [batch, n_features, seq_len] -> [batch, embed_dim, seq_len]
        x = x.permute(2, 0, 1)  # [batch, embed_dim, seq_len] -> [seq_len, batch, embed_dim]
        x = self.transformer(x)[0]  # Apply transformer and get the [CLS] token
        return x  # [batch, embed_dim]

class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024+512, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, out):
        # output = torch.cat((x, y), dim=1)
        output = self.fc_out(out)
        return output



class RGBClassifier(nn.Module):
    def __init__(self, args):
        super(RGBClassifier, self).__init__()

        n_classes = 101

        self.visual_net = resnet18(modality='visual')
        self.visual_net.load_state_dict(torch.load('/home/yake_wei/models/resnet18.pth'), strict=False)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, visual):
        B = visual.size()[0]
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        v = F.adaptive_avg_pool3d(v, 1)

        v = torch.flatten(v, 1)

        out = self.fc(v)

        return out

class FlowClassifier(nn.Module):
    def __init__(self, args):
        super(FlowClassifier, self).__init__()

        n_classes = 101

        self.flow_net = resnet18(modality='flow')
        state = torch.load('/home/yake_wei/models/resnet18.pth')
        del state['conv1.weight']
        self.flow_net.load_state_dict(state, strict=False)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, flow):
        B = flow.size()[0]
        v = self.flow_net(flow)

        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        v = F.adaptive_avg_pool3d(v, 1)

        v = torch.flatten(v, 1)

        out = self.fc(v)

        return out

class RFClassifier(nn.Module):
    def __init__(self, args):
        super(RFClassifier, self).__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'UCF101':
            n_classes = 101
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.flow_net = resnet18(modality='flow')
        state = torch.load('/home/yake_wei/models/resnet18.pth')
        del state['conv1.weight']
        self.flow_net.load_state_dict(state, strict=False)
        print('load pretrain')
        self.visual_net = resnet18(modality='visual')
        self.visual_net.load_state_dict(torch.load('/home/yake_wei/models/resnet18.pth'), strict=False)
        print('load pretrain')

        self.head = nn.Linear(1024, n_classes)
        self.head_flow = nn.Linear(512, n_classes)
        self.head_video = nn.Linear(512, n_classes)



    def forward(self, flow, visual):
        B = visual.size()[0]
        f = self.flow_net(flow)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        (_, C, H, W) = f.size()
        f = f.view(B, -1, C, H, W)
        f = f.permute(0, 2, 1, 3, 4)


        f = F.adaptive_avg_pool3d(f, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        f = torch.flatten(f, 1)
        v = torch.flatten(v, 1)

        
        out = torch.cat((f,v),1)
        out = self.head(out)

        out_flow=self.head_flow(f)
        out_video=self.head_video(v)

        return out,out_flow,out_video


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'AVMNIST':
            n_classes = 10
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))


        self.dataset = args.dataset
        self.args = args

        self.audio_net = resnet18(modality='audio')
        if args.dataset == 'AVMNIST':
            self.visual_net = resnet18(modality='image')
        else:
            self.visual_net = resnet18(modality='visual')

        self.head = nn.Linear(1024, n_classes)
        self.head_audio = nn.Linear(512, n_classes)
        self.head_video = nn.Linear(512, n_classes)



    def forward(self, audio, visual):
        # if self.dataset != 'CREMAD':
        #     visual = visual.permute(0, 2, 1, 3, 4).contiguous()
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        if self.args.dataset == 'AVMNIST':
            v = F.adaptive_avg_pool2d(v, 1)
        else:
            v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)


        out = torch.cat((a,v),1)
        out = self.head(out)

        out_audio=self.head_audio(a)
        out_video=self.head_video(v)

        return out,out_audio,out_video


class AVClassifier_OGM(nn.Module):
    def __init__(self, args):
        super(AVClassifier_OGM, self).__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))


        self.dataset = args.dataset

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        self.head = nn.Linear(1024, n_classes)
        self.head_audio = nn.Linear(512, n_classes)
        self.head_video = nn.Linear(512, n_classes)



    def forward(self, audio, visual):
        if self.dataset != 'CREMAD':
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)


        out = torch.cat((a,v),1)
        out = self.head(out)


        return a,v,out

class URFunnyClassifier(nn.Module):
    def __init__(self, args):
        super(URFunnyClassifier, self).__init__()

        # Define number of classes based on the dataset
        n_classes = args.n_classes

        if n_classes is None:
            raise NotImplementedError(f"Incorrect dataset name {self.dataset}")

        input_dim = 768  # Assumed Transformer output dimension
        # Initialize only the encoders specified in modality_combo
        self.audio_net = Transformer(81, input_dim, 4, 8)
        self.visual_net = Transformer(371, input_dim, 4, 8)
        self.text_net = Transformer(300, input_dim, 4, 8)

        self.head = nn.Linear(input_dim * 3, n_classes)
        self.head_audio = nn.Linear(input_dim, n_classes)
        self.head_video = nn.Linear(input_dim, n_classes)
        self.head_text = nn.Linear(input_dim, n_classes)


    def forward(self, audio, visual, token):
        audio_features = self.audio_net(audio)
        visual_features = self.visual_net(visual)
        text_features = self.text_net(token)

        out = torch.cat((audio_features,visual_features,text_features),1)
        out = self.head(out)

        out_audio=self.head_audio(audio_features)
        out_video=self.head_video(visual_features)
        out_text=self.head_text(text_features)

        return out,out_audio,out_video,out_text


class URFunnyAVClassifier(nn.Module):
    def __init__(self, args):
        super(URFunnyAVClassifier, self).__init__()

        # Define number of classes based on the dataset
        n_classes = args.n_classes

        if n_classes is None:
            raise NotImplementedError(f"Incorrect dataset name {self.dataset}")

        input_dim = 768  # Assumed Transformer output dimension
        # Initialize only the encoders specified in modality_combo
        self.audio_net = Transformer(81, input_dim, 4, 8)
        self.visual_net = Transformer(371, input_dim, 4, 8)

        self.head = nn.Linear(input_dim * 2, n_classes)
        self.head_audio = nn.Linear(input_dim, n_classes)
        self.head_video = nn.Linear(input_dim, n_classes)


    def forward(self, audio, visual):
        audio_features = self.audio_net(audio)
        visual_features = self.visual_net(visual)

        out = torch.cat((audio_features,visual_features),1)
        out = self.head(out)

        out_audio=self.head_audio(audio_features)
        out_video=self.head_video(visual_features)

        return out,out_audio,out_video
    
class URFunnyATClassifier(nn.Module):
    def __init__(self, args):
        super(URFunnyATClassifier, self).__init__()

        # Define number of classes based on the dataset
        n_classes = args.n_classes

        if n_classes is None:
            raise NotImplementedError(f"Incorrect dataset name {self.dataset}")

        input_dim = 768  # Assumed Transformer output dimension
        # Initialize only the encoders specified in modality_combo
        self.audio_net = Transformer(81, input_dim, 4, 8)
        self.text_net = Transformer(300, input_dim, 4, 8)

        self.head = nn.Linear(input_dim * 2, n_classes)
        self.head_audio = nn.Linear(input_dim, n_classes)
        self.head_text = nn.Linear(input_dim, n_classes)


    def forward(self, audio, token):
        audio_features = self.audio_net(audio)
        text_features = self.text_net(token)

        out = torch.cat((audio_features,text_features),1)
        out = self.head(out)

        out_audio=self.head_audio(audio_features)
        out_text=self.head_text(text_features)

        return out,out_audio,out_text
    

class URFunnyVTClassifier(nn.Module):
    def __init__(self, args):
        super(URFunnyVTClassifier, self).__init__()

        # Define number of classes based on the dataset
        n_classes = args.n_classes

        if n_classes is None:
            raise NotImplementedError(f"Incorrect dataset name {self.dataset}")

        input_dim = 768  # Assumed Transformer output dimension
        # Initialize only the encoders specified in modality_combo
        self.visual_net = Transformer(371, input_dim, 4, 8)
        self.text_net = Transformer(300, input_dim, 4, 8)

        self.head = nn.Linear(input_dim * 2, n_classes)
        self.head_video = nn.Linear(input_dim, n_classes)
        self.head_text = nn.Linear(input_dim, n_classes)


    def forward(self, visual, token):
        visual_features = self.visual_net(visual)
        text_features = self.text_net(token)

        out = torch.cat((visual_features,text_features),1)
        out = self.head(out)

        out_video=self.head_video(visual_features)
        out_text=self.head_text(text_features)

        return out,out_video,out_text


class CLIPClassifier(nn.Module):
    def __init__(self, args, image_encoder_name='ViT-B/32'):
        super(CLIPClassifier, self).__init__()
        
        self.image_encoder_name = image_encoder_name
        self.dataset = "MVSA"

        # Define number of classes based on the dataset
        n_classes = args.n_classes

        if n_classes is None:
            raise NotImplementedError(f"Incorrect dataset name {self.dataset}")

        # Load pre-trained CLIP model (text and image encoders)
        self.text_net, _ = clip.load(self.image_encoder_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.visual_net, _ = clip.load(self.image_encoder_name, device="cuda" if torch.cuda.is_available() else "cpu")

        if self.image_encoder_name == 'ViT-B/32':
            input_dim = 512
        elif self.image_encoder_name == 'RN50':
            input_dim = 1024

        self.head = nn.Linear(input_dim * 2, n_classes)
        self.head_video = nn.Linear(input_dim, n_classes)
        self.head_text = nn.Linear(input_dim, n_classes)

    def forward(self, visual, token):

        # Process text modality
        token = token.squeeze(1)
        text_features = self.text_net.encode_text(token)
        text_features = text_features.to(self.head.weight.dtype)

        # Process visual modality
        visual_features = self.visual_net.encode_image(visual)
        visual_features = visual_features.to(self.head.weight.dtype)

        out = torch.cat((visual_features,text_features),1)
        out = self.head(out)

        out_video=self.head_video(visual_features)
        out_text=self.head_text(text_features)

        return out,out_video,out_text




        
    

        
    




