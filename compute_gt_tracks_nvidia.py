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
from utils.colmap_utils import read_cameras_binary, read_points3d_binary,read_images_binary, get_colmap_camera_params, get_intrinsics_extrinsics

#-------- import cotracker -------------
from models.cotracker.utils.visualizer import Visualizer, read_video_from_path
from models.cotracker.predictor import CoTrackerPredictor

#-------- import spatialtracker -------------
from models.spatracker.predictor import SpaTrackerPredictor
from models.spatracker.utils.visualizer import Visualizer, read_video_from_path

#-------- import Depth Estimator -------------
from mde import MonoDEst


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
    # path to downsampled pcd file
    downsampled_ply: str
    
    depth_dir: str | None = None
    # directory of all the rgb images
    rgb_dir: str | None = None
    
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
    nvidia_factor: int = 2
    # video_ids: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    video_id: int = 1      # 1-12
    # factor: int = 2
    # gpu: int = 0
    # whether to visualize the support points
    vis_support: bool = True
    use_pcd: Literal["sparse", "fused", "downsampled"] = "sparse"
    skip_load_img: bool = True
    
def main(args:Config):
    fps_vis = args.fps_vis

    sparse_dir = args.sparse_dir
    colmap_images = read_images_binary(os.path.join(sparse_dir, 'images.bin'))
    colmap_point3d = read_points3d_binary(os.path.join(sparse_dir, "points3D.bin"))
    colmap_cameras = read_cameras_binary(os.path.join(sparse_dir, 'cameras.bin'))
    
    data_dir = args.data_dir
        
    cam_num = 12
    # video_ids = [i + 1 for i in range(cam_num)]
    video_ids = [args.video_id]
    
    Ks, extrins = {}, {}
    for i in range(cam_num):
        image_id = f"image{i+1}"
        K, extrin = get_intrinsics_extrinsics(colmap_images[image_id], colmap_cameras)
        Ks[image_id] = K
        extrins[image_id] = extrin
    
    fused_pcd = o3d.io.read_point_cloud(args.fused_ply)
    fused_pcd = np.asarray(fused_pcd.points)
    
    if args.use_pcd == "downsampled":
        downsampled_pcd = o3d.io.read_point_cloud(args.downsampled_ply)
        downsampled_pcd = np.asarray(downsampled_pcd.points)
        # fused_pcd = downsampled_pcd
    depth_dir = args.depth_dir
    
    # colmap_cameras = read_cameras_binary(os.path.join(sparse_dir, 'cameras.bin'))
    video_frames = {}           # {video_id: [frame_0, ...]}
    video_frame_names = {}      # {video_id: [frame_name_0, ...]}
    video_frame_depths = {}     # {video_id: [depth_0, ...]}
    for video_id in video_ids:
        video_frames[video_id] = []
        video_frame_names[video_id] = []
        video_frame_depths[video_id] = []
        img_num = len(glob.glob(os.path.join(data_dir, f"cam{video_id:02d}/*.png")))
        for i in tqdm(range(img_num)):
            # if i == 10: break
            img_filepath = os.path.join(data_dir, f"cam{video_id:02d}/{i}.png")
            image = cv2.imread(img_filepath)
            height, width = image.shape[:2]
            new_height, new_width = height // args.nvidia_factor, width // args.nvidia_factor
            image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR)
            video_frames[video_id].append(image)
            
            if depth_dir is not None:
                depth = np.load(os.path.join(depth_dir, f"{video_id}/{i}.npy"))
                video_frame_depths[video_id].append(depth)

    for video_id in video_frames.keys():
        if len(video_frame_depths[video_id]) != 0:
            video_frame_depths[video_id] = np.stack(video_frame_depths[video_id], axis=0)
            video_frame_depths[video_id] = torch.from_numpy(video_frame_depths[video_id]).float().cuda()[:,None]
        else:
            video_frame_depths[video_id] = None
            
    sparse_dir = args.sparse_dir

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
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
                video = video.cuda()
            
            cfg = edict({
                "mde_name": "zoedepth_nk"
            })
            
            MonoDEst_M = None
            depths = video_frame_depths[video_id]
            if depths is None:
                MonoDEst_O = MonoDEst(cfg)
                MonoDEst_M = MonoDEst_O.model
                MonoDEst_M.eval()
                depths = None
                
            queries = []
            queried_point3d_ids = np.array([], dtype=int)
            queried_point3d_xyzs = []
            
            image_id = f"image{video_id}"
            queried_frame = 6
            if args.use_pcd == "sparse":
                query_xys = colmap_images[image_id].xys
                point3d_ids = colmap_images[image_id].point3D_ids
                
                valid_mask = (point3d_ids > 0)
                point3d_ids = point3d_ids[valid_mask]
                query_xys = query_xys[valid_mask] / args.nvidia_factor
                
                query_txys = np.concatenate([6 * np.ones_like(query_xys[:, 0:1]), query_xys], axis=1)
                for id in point3d_ids.tolist():
                    queried_point3d_xyzs.append(colmap_point3d[id].xyz)
                queried_point3d_ids = np.concatenate([queried_point3d_ids, point3d_ids])
                queries.append(query_txys)
            elif args.use_pcd == "downsampled":
                if downsampled_pcd is None:
                    raise ValueError("Downsampled pcd is not provided")
                tracking_pts = np.concatenate([downsampled_pcd, np.ones_like(downsampled_pcd[:, 0:1])], axis=1)
                tracking_pts = tracking_pts @ extrins[image_id].T @ Ks[image_id].T
                tracking_pts = tracking_pts[:, :2] / tracking_pts[:, 2:3]
                tracking_pts /= args.nvidia_factor
                valid_mask = (tracking_pts[:, 0] >= 0) & (tracking_pts[:, 0] < W) & (tracking_pts[:, 1] >= 0) & (tracking_pts[:, 1] < H)
                tracking_pts = tracking_pts[valid_mask]
                queried_point3d_xyzs = downsampled_pcd[valid_mask]
                query_txyz = np.concatenate([queried_frame * np.ones_like(tracking_pts[:, 0:1]), tracking_pts], axis=1)
                queries.append(query_txyz)
                
                test_img = video_frames[video_id][6].copy()
                for pt in tracking_pts:
                    cv2.circle(test_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
                cv2.imwrite(f"{outdir}/query_img_video_{video_id}.png", test_img)
            else:
                raise ValueError(f"Invalid pcd type {args.use_pcd=}")
            
            queried_point3d_xyzs = np.array(queried_point3d_xyzs)
            queries = torch.from_numpy(
                np.concatenate(queries, axis=0)[None]
            ).float().cuda()
            print(f"{queries.shape=}")
            queried_point3d_xyzs_tree = cKDTree(queried_point3d_xyzs)
            fused_pcd_tree = cKDTree(fused_pcd)
            distance_0, _ = fused_pcd_tree.query(queried_point3d_xyzs)
            
            threshold = 1.2 * (np.quantile(distance_0, 0.8) + 0.01)
            
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
            batch_size = 1024 * 8
            for i in range(0, len_queries, batch_size):
                print(f"Processing {i} to {i+batch_size} queries")
                with torch.no_grad():
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
                # break
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
        np_pred_visiblity = pred_visibility[0].detach().cpu().numpy()
        np.save(os.path.join(outdir, f'{args.vid_name}_{args.model}_cam_{video_id}_visibility.npy'), np_pred_visiblity)
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
        del model, pred_tracks, pred_visibility, video_vis, vis, MonoDEst_M
        torch.cuda.empty_cache()
        # break

    
if __name__ == "__main__":
    main(tyro.cli(Config))
    