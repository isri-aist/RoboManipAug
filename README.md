# [RoboManipAug](https://github.com/isri-aist/RoboManipAug)
Data augmentation for robust imitation learning of robot manipulation

## Install
Install RoboManipBaselines with act extra as described in the following page:  
https://github.com/isri-aist/RoboManipBaselines/blob/master/doc/install.md

Install RoboManipAug:
```console
$ git clone git@github.com:isri-aist/RoboManipAug.git --recursive
$ cd RoboManipAug
$ pip install -e .
```

## Examples
### Data
Collect single data by teleoperation:
```console
$ cd robo_manip_baselines/teleop
$ python bin/TeleopMujocoUR5eInsert.py --world_idx_list 2
```

Generate environment point cloud from the hand camera RGBD images in the teleoperation data:
```console
$ cd robo_manip_aug
$ python ./bin/GenerateMergedPointCloud.py ./teleop_data/sample/MujocoUR5eInsert_base_demo.hdf5 ./env_data/MujocoUR5eInsert.pcd
```

[optional] Visualize environment point cloud with a 3D viewer:
```console
$ cd robo_manip_aug
$ python ./bin/VisualizePointCloud.py ./env_data/MujocoUR5eInsert.pcd
```

Annotate acceptable regions of end-effectors in data augmentation:
```console
$ cd robo_manip_aug
$ python ./bin/AnnotateAcceptableRegion.py ./teleop_data/sample/MujocoUR5eInsert_base_demo.hdf5 --point_cloud_path ./env_data/MujocoUR5eInsert.pcd
```

Collect additional data within the acceptable regions in the simulation:
```console
$ cd robo_manip_aug
$ python ./bin/CollectAdditionalDataMujocoUR5eInsert.py --base_demo_path ./teleop_data/sample/MujocoUR5eInsert_base_demo.hdf5 --annotation_path ./annotation_data/sample/MujocoUR5eInsert_base_demo_Annotation.pkl
```

[optional] Visualize the trajectories of the collected data with a 3D viewer:
```console
$ cd robo_manip_aug
$ python ./bin/VisualizeData3D.py ./augmented_data/sample/ --base_demo_path ./teleop_data/sample/MujocoUR5eInsert_base_demo.hdf5
```

[optional] Replay the augumented data:
```console
$ cd robo_manip_baselines/teleop
$ python ./bin/TeleopMujocoUR5eInsert.py --replay_log ../../../RoboManipAug/robo_manip_aug/augmented_data/sample/env2/MujocoUR5eInsert_base_demo_Augmented_000_00.hdf5 --replay_keys command_eef_pose_rel
```

### Policy
Train ACT policy:
```console
$ cd robo_manip_baselines/act
$ python ./bin/TrainAct.py --dataset_dir ../../../RoboManipAug/robo_manip_aug/augmented_data/sample/ --checkpoint_dir ./checkpoint/Act_MujocoUR5eInsert_RoboManipAug --state_keys --action_keys command_eef_pose_rel --camera_names hand --train_ratio 0.9 --chunk_size 1
```

Rollout ACT policy:
```console
$ cd robo_manip_baselines/act
$ python ./bin/rollout/RolloutActMujocoUR5eInsert.py --checkpoint ./checkpoint/Act_MujocoUR5eInsert_RoboManipAug/policy_last.ckpt --world_idx 2
```

### Environment
Extract images for learning 3D gaussian splatting from teleoperation data:
```console
$ cd robo_manip_aug
$ python ./bin/ExtractImagesFromData.py <path_to_hdf5> --out_dir ./env_data/MujocoUR5eInsert
```

Visualize environment point cloud with a 3D viewer:
```console
$ cd robo_manip_aug
$ python ./bin/VisualizePointCloud.py ./env_data/MujocoUR5eInsert_SuGaR.ply
```
