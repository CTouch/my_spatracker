# DEMO

``` bash
python compute_gt_tracks.py --model spatracker --data-dir hypernerf_dataset/vrig_chicken/vrig-chicken/ --sparse-dir hypernerf_dataset/vrig_chicken/vrig-chicken/colmap/dense/workspace/sparse/ --vid-name vrig_chicken --len-track 1 --depth-dir hypernerf_dataset/vrig_chicken/vrig-chicken/flow3d_preprocessed/aligned_colmap_depth/2x --fused-ply hypernerf_dataset/vrig_chicken/vrig-chicken/colmap/dense/workspace/fused.ply --factor 2 --outdir vis_result
```

# output

* `xxx_pred_track.mp4` 可视化轨迹点视频
* `xxx_tracks.npy` 保存的轨迹结果， [T, N, 3]其中[T, N, :2]表示每一帧2d图像像素上对应的像素坐标
* `xxx_fused_track_index.npy` 稠密点云和tracks关键点的对应关系，-1表示没有对应的点，可能有多个很近的点对应tracking中的同一点
* `xxx_visibility.npy` [T, N] True表示在该帧可见
