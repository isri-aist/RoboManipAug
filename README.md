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
```console
$ cd robo_manip_aug
$ python bin/AnnotateAcceptableRegion.py teleop_data/sample/MujocoUR5eInsert_env2_000.hdf5
```

```console
$ cd robo_manip_aug
$ python bin/CollectAdditionalDataMujocoUR5eInsert.py --base_demo_path ./teleop_data/MujocoUR5eInsert_env0_000.npz --annotation_path annotation_data/MujocoUR5eInsert_env0_000_Annotation.yaml
```

```console
$ cd robo_manip_aug
$ python bin/VisualizePointCloud.py env_data/MujocoUR5eCable_SuGaR.ply
```
