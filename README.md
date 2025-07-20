# [RoboManipAug](https://github.com/isri-aist/RoboManipAug)
Data augmentation for robust imitation learning of robot manipulation

## Install
Install RoboManipBaselines with the `act` extra as described [here](https://github.com/isri-aist/RoboManipBaselines/blob/master/doc/install.md#act).

Install RoboManipAug:
```console
$ git clone git@github.com:isri-aist/RoboManipAug.git --recursive
$ cd RoboManipAug
$ pip install -e .
```

> [!NOTE]
> Clone both `RoboManipBaselines` and `RoboManipAug` under the same directory, e.g.:
> ```console
> $ ls <workspace_dir>
> RoboManipBaselines  RoboManipAug
> ```

## Examples
### Data
#### Collect single data by teleoperation:
```console
# Go to the top directory of RoboManipBaselines
$ cd robo_manip_baselines
$ python bin/Teleop.py MujocoUR5eInsert --world_idx_list 2
```

https://github.com/user-attachments/assets/789415ee-1b59-48f8-bf3a-fbe5a710edea

Move the saved RMB file to `robo_manip_aug/teleop_data/sample/MujocoUR5eInsert_base_demo.rmb`.  
(The `sample` directory in the following can be an arbitrary name.)

#### Generate environment point cloud from the hand camera RGBD images in the teleoperation data:
```console
# Go to the top directory of RoboManipAug
$ cd robo_manip_aug
$ python ./bin/GenerateMergedPointCloud.py ./teleop_data/sample/MujocoUR5eInsert_base_demo.rmb ./env_data/sample/MujocoUR5eInsert.pcd
```

<img width="1960" height="1190" alt="RoboManipAug-GenerateMergedPointCloud" src="https://github.com/user-attachments/assets/8b50f7f2-f1cf-47f1-b2aa-4ef0bfecb0f4" />

#### [Optional] Visualize environment point cloud with a 3D viewer:
```console
# Go to the top directory of RoboManipAug
$ cd robo_manip_aug
$ python ./bin/VisualizePointCloud.py ./env_data/sample/MujocoUR5eInsert.pcd
```

#### Annotate acceptable regions of end-effectors in data augmentation:
```console
# Go to the top directory of RoboManipAug
$ cd robo_manip_aug
$ python ./bin/AnnotateAcceptableRegion.py ./teleop_data/sample/MujocoUR5eInsert_base_demo.rmb ./annotation_data/sample/MujocoUR5eInsert_annotation.pkl --point_cloud_path ./env_data/sample/MujocoUR5eInsert.pcd
```
When the `--load_annotation` option is specified, the already saved acceptable regions will be visualized.

<img width="1960" height="1190" alt="RoboManipAug-AnnotateAcceptableRegion" src="https://github.com/user-attachments/assets/0d2d3d0f-6941-453b-9c4b-808121f48e04" />

#### Collect augmented data within the acceptable regions in the simulation:
```console
# Go to the top directory of RoboManipAug
$ cd robo_manip_aug
$ python ./bin/CollectAugmentedData.py MujocoUR5eInsert ./teleop_data/sample/MujocoUR5eInsert_base_demo.rmb ./annotation_data/sample/MujocoUR5eInsert_annotation.pkl
```

https://github.com/user-attachments/assets/b1580334-83b7-4009-80a6-ed5e74c9ef3b

The augmented data is stored in `./augmented_data/MujocoUR5eInsert_<data_suffix>`. Rename this directory to `./augmented_data/sample/MujocoUR5eInsert`.

#### [Optional] Visualize the trajectories of the collected data with a 3D viewer:
```console
# Go to the top directory of RoboManipAug
$ cd robo_manip_aug
$ python ./bin/VisualizeData3D.py ./augmented_data/sample/MujocoUR5eInsert/ --base_demo_path ./teleop_data/sample/MujocoUR5eInsert_base_demo.rmb --point_cloud_path ./env_data/sample/MujocoUR5eInsert.pcd
```

<img width="1960" height="1190" alt="RoboManipAug-VisualizeData3D" src="https://github.com/user-attachments/assets/15287a53-6698-49df-929e-1c67d0c70ff5" />

[Optional] Plot base data and augmented data
```console
# Go to the top directory of RoboManipAug
$ cd robo_manip_aug
$ python ./misc/PlotAugmentedData.py ./augmented_data/sample/MujocoUR5eInsert/base_demo.rmb ./augmented_data/sample/MujocoUR5eInsert/region000/MujocoUR5eInsert_augmented_region000_00.rmb
```

You can plot the end-effector pose by specifying the `--data_key eef_pose` option (by default, `joint_pos` is plotted).

#### [Optional] Replay the augmented data:
```console
# Go to the top directory of RoboManipBaselines
$ cd robo_manip_baselines
$ python ./bin/Teleop.py MujocoUR5eInsert --replay_log ../../RoboManipAug/robo_manip_aug/augmented_data/sample/MujocoUR5eInsert/region000/MujocoUR5eInsert_augmented_region000_00.rmb --replay_keys command_eef_pose_rel
```

### Policy
#### Compose learning data:
```console
# Go to the top directory of RoboManipAug
$ cd robo_manip_aug
$ python ./bin/ComposeDataset.py ./augmented_data/sample/MujocoUR5eInsert ./learning_data/sample/MujocoUR5eInsert
```

By adding the `--num_data_per_region <N>` option, you can specify the number of data per region to be N (by default, all data is used).

#### Train policy:
```console
# Go to the top directory of RoboManipBaselines
$ cd robo_manip_baselines
$ python ./bin/Train.py Mlp --dataset_dir ../../RoboManipAug/robo_manip_aug/learning_data/sample/MujocoUR5eInsert/ --checkpoint_dir ./checkpoint/sample/MujocoUR5eInsert/ --state_keys --action_keys command_eef_pose_rel --camera_names hand --train_ratio 1.0 --val_ratio 0.2
```

#### Rollout policy:
```console
# Go to the top directory of RoboManipBaselines
$ cd robo_manip_baselines
$ python ./bin/Rollout.py Mlp MujocoUR5eInsert --checkpoint ./checkpoint/sample/MujocoUR5eInsert/policy_last.ckpt --world_idx 2 --world_random_scale 0.05 0.05 0.0
```

By using the option `--world_idx_list 2 2 2 2` instead of `--world_idx 2`, the rollout will be repeated four times.

### [Deprecated] Environment
#### Extract images for learning 3D gaussian splatting from teleoperation data:
```console
# Go to the top directory of RoboManipAug
$ cd robo_manip_aug
$ python ./bin/ExtractImagesFromData.py <path_to_rmb> --out_dir ./env_data/MujocoUR5eInsert
```

#### Visualize environment point cloud with a 3D viewer:
```console
# Go to the top directory of RoboManipAug
$ cd robo_manip_aug
$ python ./bin/VisualizePointCloud.py ./env_data/MujocoUR5eInsert_SuGaR.ply
```
