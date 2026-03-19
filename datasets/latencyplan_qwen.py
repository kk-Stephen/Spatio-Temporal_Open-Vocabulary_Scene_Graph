import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2

class LatencyQADataset(Dataset):
    def __init__(self, csv_path, video_path, split='train'):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['split'] == split]
        self.questions = self.data['question']
        self.answers = self.data['answer']
        self.start = self.data['start']
        self.end = self.data['end']
        # data row should have ['question', ‘latency’, 'answer', 'video_id', 'start', 'end']
        self.video_path = video_path

        self.clips = []
        # Todo: create frames

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        video = clip['video_id']
        frames = clip['frames'] #frame id list
        depth = clip['depth'] #depth frame id list

        # Todo: load frames
        images = []
        D_images = []

        prompt = ''
        message = [
            {
                'role':'user',
                'content':[
                    {
                        'type': 'video',
                        'video':'', #path list to the frames
                    },
                    {
                        'type': 'text',
                        'text':prompt,
                    },
                ],
            }
        ]

        return {
            'message': message,
            'label': self.answers[idx]
        }