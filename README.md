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
Annotate acceptable regions of end-effectors in data augmentation:
```console
$ cd robo_manip_aug
$ python ./bin/AnnotateAcceptableRegion.py ./teleop_data/sample/MujocoUR5eInsert_env2_000.hdf5
```

Collect additional data within the acceptable regions in the simulation:
```console
$ cd robo_manip_aug
$ python ./bin/CollectAdditionalDataMujocoUR5eInsert.py --base_demo_path ./teleop_data/sample/MujocoUR5eInsert_env2_000.hdf5 --annotation_path ./annotation_data/sample/MujocoUR5eInsert_env2_000_Annotation.pkl
```

Visualize the trajectories of the collected data with a 3D viewer:
```console
$ cd robo_manip_aug
$ python ./bin/VisualizeData3D.py ./augmented_data/sample/ --base_demo_path ./teleop_data/sample/MujocoUR5eInsert_env2_000.hdf5
```

Replay the augumented data:
```console
$ cd robo_manip_baselines/teleop
$ python ./bin/TeleopMujocoUR5eInsert.py --replay_log ../../../RoboManipAug/robo_manip_aug/augmented_data/sample/env2/MujocoUR5eInsert_env2_000_Augmented_000_00.hdf5 --replay_keys command_eef_pose_rel
```

Train ACT policy:
```console
$ cd robo_manip_baselines/act
$ python ./bin/TrainAct.py --dataset_dir ../../../RoboManipAug/robo_manip_aug/augmented_data/sample/ --checkpoint_dir ./checkpoint/MujocoUR5eInsert_Act_RoboManipAug --state_keys --action_keys command_eef_pose_rel --camera_names hand
```

Rollout ACT policy:
```console
$ cd robo_manip_baselines/act
$ python ./bin/rollout/RolloutActMujocoUR5eInsert.py --checkpoint ./checkpoint/MujocoUR5eInsert_Act_RoboManipAug/policy_last.ckpt --world_idx 2
```

Visualize environment point cloud with a 3D viewer:
```console
$ cd robo_manip_aug
$ python ./bin/VisualizePointCloud.py ./env_data/MujocoUR5eCable_SuGaR.ply
```
