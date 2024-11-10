
# python3 -m debugpy --wait-for-client --listen 15679 compute_gt_tracks.py \
scene_name_list=("Balloon1" "Balloon2"  "DynamicFace" "Jumping" "Playground" "Skating" "Truck" "Umbrella")
# scene_name_list=("Balloon2" "DynamicFace" "Jumping" "Playground" "Skating" "Truck" "Balloon1")
# scene_name_list=("Balloon1" "Truck" "Skating" "Playground" "Jumping" "DynamicFace" "Balloon2")
for scene_name in "${scene_name_list[@]}"; do
    python downsample_point.py data/nvidia/${scene_name}/points3D_multipleview.ply data/nvidia/${scene_name}/ds_track_pcd.ply
    video_ids_list=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")
    for video_id in "${video_ids_list[@]}"; do
        echo "Processing ${scene_name} ${video_id}...."
        python compute_gt_tracks_nvidia.py \
            --model spatracker \
            --data-dir data/nvidia/${scene_name} \
            --fused-ply data/nvidia/${scene_name}/points3D_multipleview.ply \
            --downsampled-ply data/nvidia/${scene_name}/ds_track_pcd.ply \
            --sparse-dir data/nvidia/${scene_name}/sparse_ \
            --vid-name ${scene_name} \
            --outdir vis_results/nvidia/${scene_name} \
            --len-track 1 \
            --use_pcd downsampled \
            --video-id ${video_id}
    done
done

# video_id=10
# python compute_gt_tracks_nvidia.py \
#     --model spatracker \
#     --data-dir data/nvidia/${scene_name} \
#     --fused-ply data/nvidia/${scene_name}/points3D_multipleview.ply \
#     --downsampled-ply data/nvidia/${scene_name}/ds_track_pcd.ply \
#     --sparse-dir data/nvidia/${scene_name}/sparse_ \
#     --vid-name ${scene_name} \
#     --outdir vis_results/nvidia/${scene_name} \
#     --len-track 1 \
#     --use_pcd downsampled \
#     --video-id ${video_id}