# [RoboManipAug](https://github.com/isri-aist/RoboManipAug)
Data augmentation for robust imitation learning of robot manipulation

## Examples
```console
$ cd robo_manip_aug
$ python bin/AnnotateAcceptableRegion.py teleop_data/MujocoUR5eInsert_env0_000.npz
```

```console
$ cd robo_manip_aug
$ python bin/CollectAdditionalDataMujocoUR5eInsert.py --base_demo_path ./teleop_data/MujocoUR5eInsert_env0_000.npz --annotation_path annotation_data/MujocoUR5eInsert_env0_000_Annotation.yaml
```
