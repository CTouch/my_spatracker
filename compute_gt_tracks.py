#-------- import the base packages -------------
import sys
import os
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
from base64 import b64encode
import numpy as np
from PIL import Image
import cv2
import argparse
from moviepy.editor import ImageSequenceClip
import torchvision.transforms as transforms
from tqdm import tqdm
from utils.colmap_utils import read_cameras_binary, read_points3d_binary,read_images_binary, get_colmap_camera_params

#-------- import cotracker -------------
from models.cotracker.utils.visualizer import Visualizer, read_video_from_path
from models.cotracker.predictor import CoTrackerPredictor

#-------- import spatialtracker -------------
from models.spatracker.predictor import SpaTrackerPredictor
from models.spatracker.utils.visualizer import Visualizer, read_video_from_path

#-------- import Depth Estimator -------------
from mde import MonoDEst


from utils.colmap_utils import get_colmap_camera_params, read_images_binary, read_cameras_binary

import os.path
import re
import cv2, os, glob
import tyro
from dataclasses import asdict, dataclass
import json
from typing import Literal
import cv2
import open3d as o3d

from scipy.spatial import cKDTree
@dataclass
class Config:
    data_dir: str
    # path to the sparse directory, containing images.bin, cameras.bin, points3D.bin
    sparse_dir: str
    vid_name: str
    # path to initialize pcd file
    fused_ply: str
    # depth dir
    depth_dir: str
    query_frame: int = 0
    fps_vis: int = 15
    fps:int = 1
    crop: bool = False
    crop_factor: float = 1.0
    # whether to backward the tracking
    backward: bool = False
    outdir: str = "./vis_results"    
    grid_size: int = 50
    downsample: float = 0.8
    model: Literal["cotracker", "spatracker"] = "spatracker"
    len_track:int = 10
    point_size:int = 3
    factor: int = 2
    # gpu: int = 0
    # whether to visualize the support points
    vis_support: bool = True
    
def main(args:Config):
    fps_vis = args.fps_vis

    ## set input
    # root_dir = args.root
    # vid_dir = os.path.join(root_dir, args.vid_name + '.mp4')
    # seg_dir = os.path.join(root_dir, args.vid_name + '.png')
    data_dir = args.data_dir
    datasetjson = json.load(open(os.path.join(data_dir, "dataset.json")))               # TODO only hypernerf dataset
    metajson = json.load(open(os.path.join(data_dir, "metadata.json")))                 # TODO only hypernerf dataset
    rgb_dir = os.path.join(data_dir, f"rgb/{args.factor}x")
    # fused_pcd = o3d.io.read_point_cloud(os.path.join(data_dir, f"colmap/dense/workspace/fused.ply"))
    fused_pcd = o3d.io.read_point_cloud(args.fused_ply)
    fused_pcd = np.asarray(fused_pcd.points)
    # depth_dir = os.path.join(data_dir, f"flow3d_preprocessed/aligned_colmap_depth/{args.factor}x")
    depth_dir = args.depth_dir
    frame_names = datasetjson["ids"]
    
    sparse_dir = args.sparse_dir
    colmap_images = read_images_binary(os.path.join(sparse_dir, 'images.bin'))
    colmap_point3d = read_points3d_binary(os.path.join(sparse_dir, "points3D.bin"))
    video_ids = ["train_ids"]
    video_frames = {}           # {video_id: [frame_0, ...]}
    video_frame_names = {}      # {video_id: [frame_name_0, ...]}
    video_frame_depths = {}     # {video_id: [depth_0, ...]}
    for video_id in video_ids:
        video_frames[video_id] = []
        video_frame_names[video_id] = []
        video_frame_depths[video_id] = []
        for frame_name in datasetjson[f"{video_id}"]:
            image = cv2.imread(os.path.join(rgb_dir, f"{frame_name}.png"))
            image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            depth = np.load(os.path.join(depth_dir, f"{frame_name}.npy"))
            video_frames[video_id].append(image)
            video_frame_names[video_id].append(frame_name)
            video_frame_depths[video_id].append(depth)
    # for frame_name in tqdm(frame_names, desc="Load dataset"):
    #     meta = metajson[frame_name]
    #     camera_id = meta["camera_id"]
    #     if not (camera_id in video_frames):
    #         video_frames[camera_id] = []
    #         video_frame_names[camera_id] = []
    #         video_frame_depths[camera_id] = []
    #     image = cv2.imread(os.path.join(rgb_dir, f"{frame_name}.png"))
    #     image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #     depth = np.load(os.path.join(depth_dir, f"{frame_name}.npy"))
    #     video_frames[camera_id].append(image)
    #     video_frame_names[camera_id].append(frame_name)
    #     video_frame_depths[camera_id].append(depth)

    for video_id in video_frames.keys():
        video_frame_depths[video_id] = np.stack(video_frame_depths[video_id], axis=0)
        video_frame_depths[video_id] = torch.from_numpy(video_frame_depths[video_id]).float().cuda()[:,None]

    sparse_dir = args.sparse_dir
    colmap_images = read_images_binary(os.path.join(sparse_dir, 'images.bin'))

    seg_dir = ""
    outdir = args.outdir
    os.path.exists(outdir) or os.makedirs(outdir)
    
    # set the paras
    grid_size = args.grid_size
    model_type = args.model
    downsample = args.downsample
    # # set the gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    transform = transforms.Compose([
        transforms.CenterCrop((int(384*args.crop_factor),
                                int(512*args.crop_factor))),  
    ])
    
    for video_id in video_frames.keys():
        # video = np.concatenate(camera_images[camera_id], axis=0)
        video = np.stack(video_frames[video_id], axis=0)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        _, T, _, H, W = video.shape
        if os.path.exists(seg_dir):
            segm_mask = np.array(Image.open(seg_dir))
        else:
            segm_mask = np.ones((H, W), dtype=np.uint8)
            print("No segmentation mask provided. Computing tracks it in whole image.")
        if len(segm_mask.shape)==3:
            segm_mask = (segm_mask[..., :3].mean(axis=-1)>0).astype(np.uint8)    
        segm_mask = cv2.resize(segm_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        if args.crop:
            video = transform(video)
            segm_mask = transform(torch.from_numpy(segm_mask[None, None]))[0,0].numpy()
        _, _, _, H, W = video.shape
        # adjust the downsample factor
        if H > W:
            downsample = max(downsample, 640//H)
        elif H < W:
            downsample = max(downsample, 960//W)
        else:
            downsample = max(downsample, 640//H)

        video = F.interpolate(video[0], scale_factor=downsample,
                            mode='bilinear', align_corners=True)[None]
        
        vidLen = video.shape[1]
        idx = torch.range(0, vidLen-1, args.fps).long()
        video=video[:, idx]
        # save the first image
        img0 = video[0,0].permute(1,2,0).detach().cpu().numpy()


        cv2.imwrite(os.path.join(outdir, f'{args.vid_name}_ref.png'), img0[:,:,::-1])
        cv2.imwrite(os.path.join(outdir, f'{args.vid_name}_seg.png'), segm_mask*255)

        if args.model == "cotracker":
            model = CoTrackerPredictor(
                checkpoint=os.path.join(
                    './checkpoints/cotracker_pretrain/cotracker_stride_4_wind_8.pth'
                )
            )
            if torch.cuda.is_available():
                model = model.cuda()
                video = video.cuda()
            pred_tracks, pred_visibility = model(video, 
                                                grid_size=grid_size,
                                                backward_tracking=False,
                                                segm_mask=torch.from_numpy(segm_mask)[None, None])
            
            vis = Visualizer(save_dir=outdir, grayscale=True, 
                            fps=fps_vis, pad_value=0, tracks_leave_trace=args.len_track)
            video_vis=vis.visualize(video=video, tracks=pred_tracks,
                                    visibility=pred_visibility, filename=args.vid_name+"_cotracker")
        elif args.model == "spatracker":
            S_lenth = 12       # [8, 12, 16] choose one you want
            model = SpaTrackerPredictor(
            checkpoint=os.path.join(
                './checkpoints/spaT_final.pth',
                ),
                interp_shape = (384, 512),
                seq_length = S_lenth
            )
            if torch.cuda.is_available():
                model = model.cuda()
                video = video.cuda()
            
            cfg = edict({
                "mde_name": "zoedepth_nk"
            })
            
            MonoDEst_M = None
            # DEPTH_DIR = os.path.join(root_dir, args.vid_name)
            # DEPTH_DIR = args.depth_dir
            # assert os.path.exists(DEPTH_DIR), "Please provide the depth maps in {DEPTH_DIR}"
            # depths = []
            # for dir_i in sorted(os.listdir(DEPTH_DIR)):
            #     depth = np.load(os.path.join(DEPTH_DIR, dir_i))
            #     depths.append(depth)
            depths = video_frame_depths[video_id]
            # depths = torch.from_numpy(depths).float().cuda()[:,None]
            
            # if args.rgbd:

            # else:
            #     MonoDEst_O = MonoDEst(cfg)
            #     MonoDEst_M = MonoDEst_O.model
            #     MonoDEst_M.eval()
            #     depths = None

            queries = []
            queried_point3d_ids = np.array([], dtype=int)
            queried_point3d_xyzs = []
            for i, frame_name in enumerate(video_frame_names[video_id]):
                query_xys = colmap_images[frame_name].xys
                point3d_ids = colmap_images[frame_name].point3D_ids
                
                valid_mask = (point3d_ids > 0) & (~np.isin(point3d_ids, queried_point3d_ids))
                point3d_ids = point3d_ids[valid_mask]
                query_xys = query_xys[valid_mask]
                
                # new_mask = ~np.isin(point3d_ids, queried_point3d_ids)
                # query_xys = query_xys[new_mask]
                query_txys = np.concatenate([i * np.ones_like(query_xys[:, 0:1]), query_xys], axis=1)
                for id in point3d_ids.tolist():
                    queried_point3d_xyzs.append(colmap_point3d[id].xyz)
                queried_point3d_ids = np.concatenate([queried_point3d_ids, point3d_ids])
                queries.append(query_txys)
            queried_point3d_xyzs = np.array(queried_point3d_xyzs)
            queries = torch.from_numpy(
                np.concatenate(queries, axis=0)[None]
            ).float().cuda()
            print(f"{queries.shape=}")
            queried_point3d_xyzs_tree = cKDTree(queried_point3d_xyzs)
            fused_pcd_tree = cKDTree(fused_pcd)
            distance_0, _ = fused_pcd_tree.query(queried_point3d_xyzs)
            
            threshold = 1.2 * np.quantile(distance_0, 0.8)
            
            distance, idx = queried_point3d_xyzs_tree.query(fused_pcd)
            mask = distance < threshold
            fused_track_index = np.ones_like(fused_pcd[:, 0], dtype=int) * -1
            fused_track_index[mask] = idx[mask]
            np.save(f"{outdir}/{args.vid_name}_cam_{video_id}_fused_track_index.npy", fused_track_index)
            
            # if args.query_2d_points is not None:
            #     filepath = args.query_2d_points
            #     if os.path.exists(filepath) and filepath.endswith('.npy'):
            #         xys = np.load(filepath)
            #         queries = torch.from_numpy(xys).float().cuda()
            #         ts = torch.ones_like(queries[..., 0:1]) * args.query_frame
            #         queries = torch.cat([ts, queries], dim=-1)[None, ...]
            #         print(f"{queries.shape=}")
                    
            #     else:
            #         raise ValueError("Invalid query_2d_points file path")
            len_queries = queries.shape[1]
            batch_size = 10000
            for i in range(0, len_queries, batch_size):
                print(f"Processing {i} to {i+batch_size} queries")
                pred_tracks, pred_visibility, T_Firsts = (
                    model(video, video_depth=depths,
                    grid_size=grid_size, backward_tracking=args.backward,
                    depth_predictor=MonoDEst_M, grid_query_frame=args.query_frame,
                    segm_mask=torch.from_numpy(segm_mask)[None, None], wind_length=S_lenth,
                    queries=queries[:, i:i+batch_size]))
                if i == 0:
                    all_pred_tracks = pred_tracks
                    all_pred_visibility = pred_visibility
                else:
                    all_pred_tracks = torch.concatenate([all_pred_tracks, pred_tracks], dim=2)
                    all_pred_visibility = torch.concatenate([all_pred_visibility, pred_visibility], dim=2)
            pred_tracks = all_pred_tracks
            pred_visibility = all_pred_visibility
            
            # pred_tracks, pred_visibility, T_Firsts = (
            #                                 model(video, video_depth=depths,
            #                                 grid_size=grid_size, backward_tracking=args.backward,
            #                                 depth_predictor=MonoDEst_M, grid_query_frame=args.query_frame,
            #                                 segm_mask=torch.from_numpy(segm_mask)[None, None], wind_length=S_lenth,
            #                                 queries=queries))
            
            vis = Visualizer(save_dir=outdir, grayscale=True, 
                                fps=fps_vis, pad_value=0, linewidth=args.point_size,
                                tracks_leave_trace=args.len_track)
            # msk_query = (T_Firsts == args.query_frame)
            # visualize the all points
            if args.vis_support:
                video_vis = vis.visualize(video=video, tracks=pred_tracks[..., :2],
                                        visibility=pred_visibility,
                                        filename=args.vid_name+ f"_spatracker_cam_{video_id}")
            else:
                pred_tracks = pred_tracks[:,:,msk_query.squeeze()]
                pred_visibility = pred_visibility[:,:,msk_query.squeeze()]
                video_vis = vis.visualize(video=video, tracks=pred_tracks[..., :2],
                                        visibility=pred_visibility,
                                        filename=args.vid_name+ f"_spatracker_cam_{video_id}")

        # vis the first queried video
        img0 = video_vis[0,0].permute(1,2,0).detach().cpu().numpy()
        cv2.imwrite(os.path.join(outdir, f'{args.vid_name}_ref_query.png'), img0[:,:,::-1])
        # save the tracks
        tracks_vis = pred_tracks[0].detach().cpu().numpy()
        np.save(os.path.join(outdir, f'{args.vid_name}_{args.model}_cam_{video_id}_tracks.npy'), tracks_vis)
        # save the video
        wide_list = list(video.unbind(1))
        wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
        clip = ImageSequenceClip(wide_list, fps=60)
        save_path = os.path.join(outdir, f'{args.vid_name}_cam_{video_id}_vid.mp4')
        clip.write_videofile(save_path, codec="libx264", fps=25, logger=None)
        print(f"Original Video saved to {save_path}")

        T = pred_tracks[0].shape[0]
        # save the 3d trajectories
        xys = pred_tracks[0].cpu().numpy()   # T x N x 3
        intr = np.array([[W, 0.0, W//2],
                        [0.0, W, H//2],
                        [0.0, 0.0, 1.0]])
        xyztVis = xys.copy()
        xyztVis[..., 2] = 1.0

        xyztVis = np.linalg.inv(intr[None, ...]) @ xyztVis.reshape(-1, 3, 1) # (TN) 3 1
        xyztVis = xyztVis.reshape(T, -1, 3) # T N 3
        xyztVis[..., 2] *= xys[..., 2]

        pred_tracks2d = pred_tracks[0][:, :, :2]
        S1, N1, _ = pred_tracks2d.shape
        video2d = video[0] # T C H W
        H1, W1 = video[0].shape[-2:] 
        pred_tracks2dNm = pred_tracks2d.clone()
        pred_tracks2dNm[..., 0] = 2*(pred_tracks2dNm[..., 0] / W1 - 0.5)
        pred_tracks2dNm[..., 1] = 2*(pred_tracks2dNm[..., 1] / H1 - 0.5)
        color_interp = torch.nn.functional.grid_sample(video2d, pred_tracks2dNm[:,:,None,:],
                                                        align_corners=True)

        color_interp = color_interp[:, :, :, 0].permute(0,2,1).cpu().numpy().astype(np.uint8)
        colored_pts = np.concatenate([xyztVis, color_interp], axis=-1)
        np.save(f'{outdir}/{args.vid_name}_cam_{video_id}_3d.npy', colored_pts)

        print(f"3d colored tracks to {outdir}/{args.vid_name}_cam_{video_id}_3d.npy")
        break

    
if __name__ == "__main__":
    main(tyro.cli(Config))
    