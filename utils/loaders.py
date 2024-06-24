import glob
from abc import ABC
import pandas as pd
from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger
import numpy as np 
import math
import torch
from .video_record import VideoRecord


class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        split: str (D1, D2 or D3)
        modalities: list(str, str, ...)
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat

        if self.load_feat:
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                model_features = pd.DataFrame(pd.read_pickle(os.path.join("extracted_features",
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                          pickle_name))['features'])[["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")

    def _get_train_indices(self, record, modality='RGB'):
        ##################################################################
        # TODO: implement sampling for training mode                     #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################
        """
        Samples frame indices for training, supporting both dense and uniform sampling.
        
        Args:
            record (EpicVideoRecord): An object containing metadata for a video clip.
            modality (str): The type of modality (e.g., RGB) to consider.
            
        Returns:
            np.ndarray: Indices of frames to sample.
        """
        # Initialize variables for clarity
        num_frames = record.num_frames[modality]
        num_samples = self.num_frames_per_clip[modality] * self.num_clips
        
        if self.dense_sampling[modality]:
            # Dense sampling logic
            center_frames = np.linspace(0, num_frames, self.num_clips + 2, dtype=np.int32)[1:-1]
            indices = []
            for center in center_frames:
                start = max(0, center - math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride))
                end = min(num_frames, center + math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride))
                indices.extend(range(start, end, self.stride))
                
            # Handle case where we have fewer frames than needed
            if len(indices) < num_samples:
                indices += [indices[-1]] * (num_samples - len(indices))
        else:
            # Uniform sampling logic
            if num_frames >= num_samples:
                stride = max(1, num_frames // num_samples)
                indices = np.arange(0, stride * num_samples, stride)
            else:
                # When there are fewer frames than needed, repeat the last frame index
                indices = np.arange(0, num_frames)
                additional_indices = [num_frames - 1] * (num_samples - len(indices))
                indices = np.concatenate((indices, additional_indices))

        # Ensure indices are within bounds
        indices = np.clip(indices, 0, num_frames - 1)
        
        return indices.astype(int)

    def _get_val_indices(self, record, modality):
        ##################################################################
        # TODO: implement sampling for testing mode                      #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################
        """
        Samples frame indices for validation/testing, supporting both dense and uniform sampling.
        
        Args:
            record (EpicVideoRecord): An object containing metadata for a video clip.
            modality (str): The type of modality (e.g., RGB) to consider.
            
        Returns:
            np.ndarray: Indices of frames to sample.
        """
        # Initialize variables for clarity
        num_frames = record.num_frames[modality]
        num_samples = self.num_frames_per_clip[modality] * self.num_clips
        
        if self.dense_sampling[modality]:
            # Dense sampling logic
            center_frames = np.linspace(0, num_frames, self.num_clips + 2, dtype=np.int32)[1:-1]
            indices = []
            for center in center_frames:
                start = max(0, center - math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride))
                end = min(num_frames, center + math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride))
                indices.extend(range(start, end, self.stride))
                
            # Handle case where we have fewer frames than needed
            if len(indices) < num_samples:
                indices += [indices[-1]] * (num_samples - len(indices))
        else:
            # Uniform sampling logic
            if num_frames >= num_samples:
                stride = max(1, num_frames // num_samples)
                indices = np.arange(0, stride * num_samples, stride)
            else:
                # When there are fewer frames than needed, repeat the last frame index
                indices = np.arange(0, num_frames)
                additional_indices = [num_frames - 1] * (num_samples - len(indices))
                indices = np.concatenate((indices, additional_indices))

        # Ensure indices are within bounds
        indices = np.clip(indices, 0, num_frames - 1)
        
        return indices.astype(int)

    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]

        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = sample_row["features_" + m].values[0]
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p)
            images.extend(frame)
        # finally, all the transformations are applied
        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added

            idx_untrimmed = record.start_frame + idx
            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, record.untrimmed_video_name, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)

class ActionNetRecord(VideoRecord):
    def __init__(self, tup, dataset_conf):
        self._index = str(tup[0])
        self._series = tup[1]
        self.dataset_conf = dataset_conf

    @property
    def start_frame(self):
        return self._series['start_frame']

    @property
    def end_frame(self):
        return self._series['stop_frame']

    @property
    def uid(self):
        return self._series['id']

    @property
    def untrimmed_video_name(self):
        return self._series['subject_id']
        
    @property
    def label(self):
        return torch.tensor(self._series['description'], dtype=torch.long)

    @property
    def myo_left_readings(self):
        return self._series['myo_left_readings']
    
    @property
    def myo_right_readings(self):
        return self._series['myo_right_readings']
    
    @property
    def num_frames(self):
        return {'RGB': self.end_frame - self.start_frame,
                'EMG': len(self._series['myo_right_readings']) }
    @property
    def get_indices(self):
        return self._series['index_frames']

class ActionNetDataset(data.Dataset, ABC):

    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs) -> None:
        """
        - split: ActionNet if modality is EMG or S04 if RGB
        - modalities can be RGB(not implemented yet) and EMG data
        - mode is a string (train, test)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        super().__init__()

        self.modalities = modalities  # considered modalities (ex. [RGB,EMG])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        self.require_spectrogram = kwargs.get('require_spectrogram', False)
        
        if self.mode == "train":
            pickle_name = "ActionNet_EMG_train.pkl"
        else:
            pickle_name = "ActionNet_EMG_test.pkl"
        
        raw_data = pd.read_pickle(os.path.join(dataset_conf.annotations_path, pickle_name))

        if split == 'S04':
          raw_data = raw_data[raw_data['subject_id'] == 'S04']


        self.list_file = raw_data
        #print(f'list_val_load: {self.list_file}, add: {os.path.join(self.dataset_conf.annotations_path, pickle_name)}')
        logger.info(f"Dataloader for {split} - {self.mode} with {len(self.list_file)} samples generated")
        self.video_list = [ ActionNetRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        
        self.transform = transform
        self.load_feat = load_feat

        if self.load_feat:
            if self.mode == "train":
              pickle_name = "ActionNet_train.pkl"
            else:
                pickle_name = "ActionNet_test.pkl"

            self.model_features = None
            for m in self.modalities:
                
                model_features = pd.DataFrame(pd.read_pickle(os.path.join(self.dataset_conf[m].data_path, 
                  pickle_name))['features'])[["id", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="id")
            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="id")
    
    def __getitem__(self, index):
        
        frames = {}
        label = None
        record = self.video_list[index]


        ##NB since for VAE we always use self.load_feat=True it could make sense to remove ther rest of the code
        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["id"] == int(record.uid)]
            if len(sample_row) > 1:
              print(record.uid)
              print(self.model_features[self.model_features["id"] == int(record.uid)])
            assert len(sample_row) == 1, f'for {record.uid} got {sample_row}'
            for m in self.modalities:
                sample[m] = torch.Tensor(sample_row["features_" + m].values[0])

            if self.additional_info:
                return sample, sample_row['description'].values, record.untrimmed_video_name, record.uid
            else:
                return sample, sample_row['description'].values
        

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
          segment_indices[modality] = record.get_indices

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img
       

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        #logger.info(f'nel_get : {indices}')
        if modality == 'RGB':
            images = list()
            for frame_index in indices:
                # here the frame is loaded in memory
                frame = self._load_data(modality, record, frame_index)
                images.extend(frame)
            # finally, all the transformations are applied
            process_data = self.transform[modality](images)
            return process_data, record.label
            
    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path

        if modality == 'RGB':
            # here the offset for the starting index of the sample is added
            img = Image.open(os.path.join(data_path, 'frame_' + str(idx).zfill(10) + '.jpg')).convert('RGB')
            return [img]
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
    
        return len(self.video_list)

