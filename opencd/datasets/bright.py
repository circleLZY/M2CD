# Copyright (c) Open-CD. All rights reserved.
from opencd.registry import DATASETS
from .basecddataset import _BaseCDDataset


@DATASETS.register_module()
class BRIGHT_Dataset(_BaseCDDataset):
    """BRIGHT dataset for building damage assessment.
    
    The dataset contains pairs of images (before and after) and a label map.
    The label map has the following pixel values:
        - 0: background
        - 1: intact building
        - 2: damaged building
        - 3: destroyed building
    """
    METAINFO = dict(
        classes=('background', 'intact', 'damaged', 'destroyed'),
        palette=[[255, 255, 255], 
                 [70, 181, 121], 
                 [228, 189, 139], 
                 [182, 70, 69]]
    )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 format_seg_map=None,  # No need to format to binary
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            **kwargs
        )


@DATASETS.register_module()
class MultiTeacherBrightDataset(BRIGHT_Dataset):
    def __init__(self, 
                 data_root=None,
                 data_prefix=None,
                 pipeline=None,
                 
                 intact_data_prefix=None,
                 damaged_data_prefix=None,
                 destroyed_data_prefix=None):
        super().__init__(data_root=data_root, data_prefix=data_prefix, pipeline=pipeline)
        
        self.intact_dataset = BRIGHT_Dataset(data_root=data_root, data_prefix=intact_data_prefix, pipeline=pipeline) if intact_data_prefix else None
        self.damaged_dataset = BRIGHT_Dataset(data_root=data_root, data_prefix=damaged_data_prefix, pipeline=pipeline) if damaged_data_prefix else None
        self.destroyed_dataset = BRIGHT_Dataset(data_root=data_root, data_prefix=destroyed_data_prefix, pipeline=pipeline) if destroyed_data_prefix else None
    

    def __getitem__(self, idx):
        if not self.test_mode:
            if self.intact_dataset is None or self.damaged_dataset is None or self.destroyed_dataset is None:
                return super().__getitem__(idx)
            else:
            # 初始化结果字典
                results = []
            # 按顺序处理每个教师数据
                for dataset in [self.intact_dataset, self.damaged_dataset, self.destroyed_dataset]:
                    data = dataset[idx % len(dataset)]  # 使用父类的__getitem__
                    results.append(data)
                return results
        else:
            return super().__getitem__(idx)  # 测试模式使用原始逻辑

