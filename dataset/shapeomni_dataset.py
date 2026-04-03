import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle as pkl


def pc_norm(pc):
    """pc: NxC, return NxC"""
    """
    pc: Nx3 array
    This functions normalizes a point cloud to fit within a unit sphere.
    It first calculates the centroid of the point cloud and then subtracts
    it from all points before scaling all points to fit within a unit sphere.
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class PointImageVoxelDataset(Dataset):
    def __init__(self, ann_paths=[], image_root_dir=""):
        self.all_data = []
        print(f"Loading data from {len(ann_paths)} files...")
        for ann_path in ann_paths:
            if os.path.exists(ann_path):
                # map_location='cpu' 节省显存
                self.all_data.extend(pkl.load(open(ann_path, "rb")))
            else:
                print(f"Warning: {ann_path} does not exist.")
        print(f"Total samples: {len(self.all_data)}")
        self.image_root_dir = image_root_dir

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        data = self.all_data[index]
        
        voxel_path = data["voxel_path"]
        voxel_np = np.load(voxel_path, allow_pickle=True)["voxel"]
        
        # --- 1. Image 处理 (只读 PIL，不处理) ---
        image_path = data.get("image_path", None)
        # 路径拼接
        if image_path and not os.path.isabs(image_path) and self.image_root_dir:
            image_path = os.path.join(self.image_root_dir, image_path)
            
        try:
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                # 异常保底：黑色小图
                image = Image.new('RGB', (100, 100), (0, 0, 0))
        except Exception:
            image = Image.new('RGB', (100, 100), (0, 0, 0))


        if isinstance(voxel_np, np.ndarray):
            voxel = torch.from_numpy(voxel_np)
            # 确保维度 [C, D, H, W] -> [1, 64, 64, 64]
        # if voxel.dim() == 3: 
        #     voxel = voxel.unsqueeze(0)

        # --- 3. 文本与其他字段 ---
        answer = data.get("ground_truth", "")
        shape_id = data.get("shape_id", "")
        point = torch.tensor(pc_norm(data["raw_point"])).float()
        masks = torch.tensor(data["GT"]).float().permute(1, 0).unsqueeze(0)
        # Label 处理逻辑 (保持一致)
        if "label" in data:
             label = data["label"]
        elif "affordance_label" in data:
            label = data["affordance_label"]
        else:
            label = data.get("semantic_class", "")

        return {
            "question": data.get("question", ""),
            "points": point,
            "image": image,
            "voxel": voxel,
            "answer": answer,
            "shape_id": shape_id,
            "label": label,
            "semantic_class": data.get("semantic_class", ""),
            "masks": masks,
        }

    def collate(self, samples):
        question = []
        image = []
        voxel = []
        answer = []
        shape_id = []
        label = []
        semantic_class = []
        masks = []
        points = []

        for sample in samples:
            question.append(sample["question"])
            image.append(sample["image"])
            voxel.append(sample["voxel"])
            answer.append(sample["answer"])
            shape_id.append(sample["shape_id"])
            label.append(sample["label"])
            semantic_class.append(sample["semantic_class"])
            masks.append(sample["masks"])
            points.append(sample["points"])

        # 对于 Voxel，通常需要堆叠成 Tensor 以便打印 shape 或输入模型
        # Image 保持为 List[PIL]，Text 保持为 List[str]
        return {
            "question": question,
            "image": image,                 # List[PIL.Image]
            "voxel": torch.stack(voxel, 0), # Tensor [B, 1, 64, 64, 64]
            "answer": answer,
            "shape_id": shape_id,
            "label": label,
            "semantic_class": semantic_class,
            "masks": masks,
            "points": points,
        }

if __name__ == "__main__":
    # 配置
    DATA_PATH = "/hard_data1/user_dataset/zhangshaolong_dataset/3D_ADLLM_DATA/data_train_meta_idx1.pt"
    
    # 1. Dataset
    dataset = PointImageVoxelDataset(ann_paths=[DATA_PATH])
    
    # 2. DataLoader
    # 关键点：必须传入 collate_fn=dataset.collate
    # 否则默认的 default_collate 会尝试将 PIL Image 转为 Tensor 并报错
    loader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=dataset.collate 
    )
    
    print("Start iterating...")
    for batch in loader:
        print("-" * 30)
        print(f"Batch Size: {len(batch['shape_id'])}")
        
        # 图像是 List[PIL]，没被处理过
        print(f"Images type: {type(batch['image'])}") 
        if len(batch['image']) > 0:
            print(f"First image size: {batch['image'][0].size}")
        
        # 体素是 Tensor (在 collate 中做了 stack)
        print(f"Voxels shape: {batch['voxel'].shape}") # Expected: [4, 1, 64, 64, 64]
        
        # 文本是 List
        print(f"Question example: {batch['question'][0]}")
        
        break