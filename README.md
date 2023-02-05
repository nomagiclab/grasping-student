# Two-finger imitation learning – cables grasping report

## Overview

This repository contains implementation of two-finger grasping pipeline,
with grasping model learned through imitation. 

The robot code is designed with deformables in mind, but one can change the settings in robot code configuration for other purposes.

It may be used for imitation learning evaluation on new items, to determine whether the robot is capable to grasp particular set of items.
It should achieve reasonable GSR after using only few hundred of examples.

The repository contains:
* Camera (Realsense D415) code
* Imitation learning grasping model implementation and training
* Imitation learning dataset
* Dataset collection and annotation tool
* Model outputs and inputs visualization for debugging purpose
* Real robot (UR5e) grasping code
* Pipeline, which uses all of the following to evaluate ability of the robot to handle particular set of items

## Quickstart
Install the requirements:
```sh
pip3 install -r requirements.txt
```

Download the imitation learning dataset:
```sh
cd dataset
python3 download_dataset.py
```

Download the trained model:
```sh
gdown --id "1f5Qy3KqYn7rcPz0RYNkiq0Py-9fZDXGO"
```

Train the grasping model through imitation:
```sh
python3 model/imitation_learning.py
```

Run the grasping pipeline: (you should edit the script passing into robot code configuration ip of your robot and description of the grasping workspace along with editing `robot/camera_pose.txt`)
```sh
python3 evaluate_grasping.py
```

Run the dataset collection pipeline:
```sh
python3 collect_dataset.py
```


## Results
We achieve about 88.5% grasp success rate. We evaluated our model on 200 grasps.
Random pick with positive depth baseline achieved 40% grasp success rate, random picks in our framework often resulted in protective stops,
which cannot be said about our approach.

### Grasping videos
We provide qualitative demonstration of the grasping system learned through imitation:

[grasping demo 50x speedup](https://youtu.be/nlbr8y2aDt4)

[grasping demo 5x speedup](https://youtu.be/huOLSlsx93k)

[grasping demo raw](https://drive.google.com/file/d/1TbQMPu1pMMoSIsnxxguPEwlHHib6QX9c/view?usp=sharing)

![grasping gif](visualizations/grasping.gif)


### Visualizations
The pipeline during the inference visualizes the inputs to neural networks along with its outputs per angle
and different statistics regarding it.

The visualization of the inputs and outputs is useful for model inspection and debugging.

The affordance outputs and inputs from evaluation for manual inspection are saved in `visualizations/affordances` and `visualizations/inputs`.

We assume grasp in this way: |----| in visualizations for different angles. 

[//]: # (![inputs gif]&#40;visualizations/inputs.gif&#41;)

![affordances gif](visualizations/affordances.gif)

## Proposed approach

### High-level overview

The highlevel loop of the two-finger grasping pipeline goes as follows: 
1. Realsense camera makes the RGBD image
2. Using camera intrinsics and camera pose in robot coordinates we compute the pointcloud
3. Using workspace limits we filter out the points in the pointcloud and project it to get the top-down view
4. We put the projected depth image into the neural network to get the (x, y, angle) grasping prediction
5. We attempt the top-down grasp at (x, y, depth at (x, y), angle)
6. We close the gripper until force, we move gripper the tcp up, read the sensors to determine whether we have grasped something

### Model inference
1. As the input to the model we put only depth image (we discard the rgb component)
2. We put into the network images by angles in [0; pi) discretized by grid of size 16
3. As a backbone we use fcn renset50
4. We put affordance scores for each angle and (x, y) into the softmax
5. We zero probabilities for <= 0.95 percentile and further normalize
6. We sample according to probabilities on (x, y, angle) to overcome potential looping behaviour

### Model training
1. As the input to the model we put only depth image (we discard the rgb component)
2. We randomly augment the image by random SE(2) transformation (random translation and rotation) – remember to also transform ground truth
3. We also apply slight random shear, scaling, gaussian noise on depth and noise on angles
4. We put into the network two images – one original and one rotated randomly, but not to close (we have arbitrary offset)
5. As a backbone we use fcn resnet50 
6. We put affordance scores for each of two angles and (x, y) into the softmax
7. We calculate and minimize binary cross entropy between output on and ground truth label in a point
8. We use AdamW optimizer with lr=1e-4, weight_decay=1e-5, max_epochs=10, cosine lr schedule up to 2e-6

To inspect the exemplary training curves, please run:\
`tensorboard --logdir model`

![affordances gif](visualizations/val-affordance.png)
![affordances gif](visualizations/val-distance.png)
![affordances gif](visualizations/val-ranking.png)

We track following metrics, they correspond to good imitation:
* the average score (affordance) in the points selected by human in the imitation learning dataset
* the expected (weighted by probabilites of selection) distance to the human selected grasp-point
* the ranking by scores of human-selected points

## Dataset
In order to train grasping model through imitation we collected small dataset of about 400 grasp attempts made by human annotator,
which were additionally verified by robotic grasping pipeline for grasp success detection.

Remember that both positive and negative (about 20%) examples were used to train the grasping model.

[google drive link](https://drive.google.com/drive/folders/1kquapAVTdhtdzc1czy5rcfBrMtXWzWIw?usp=sharing)

### Dataset description

Dataset contains pickle files, each containing trajectory, which is list of grasp attempt.
Each grasp attempt consists at least of:
* raw rgb image from realsense
* raw depth image from realsense
* projected heightmap image
* camera intrinsics
* camera pase in robot space
* workspace description
* max number of rotations
* location of grasp in (x, y, angle) form

It has the following format dictified:
```python
class GraspAttempt(NamedTuple):
    heightmap: Union[HeightMapImage, List[HeightMapImage]]
    raw_heightmap: Union[HeightMapImage, List[HeightMapImage]]

    rgb: np.ndarray
    depth: np.ndarray

    segmentation: Union[Optional[torch.Tensor], List[Optional[torch.Tensor]]]
    grasping_index: Union[GraspingIndex, List[GraspingIndex]]
    grasping_point: Union[GraspingPoint, List[GraspingPoint]]
    successful: Union[bool, List[bool]]

    camera_intrinsics: Optional[np.ndarray] = None
    camera_extrinsics: Optional[np.ndarray] = None

    rgb_normalization: Optional[Tuple[List[float], List[float]]] = None
    depth_normalization: Optional[Tuple[float, float]] = None

    num_rotations: Optional[int] = None

class GraspingIndex(NamedTuple):
    angle_index: Union[int, List[int]]
    row: Union[int, List[int]]
    col: Union[int, List[int]]


class GraspingPoint(NamedTuple):
    angle: Union[float, List[float]]
    x: Union[float, List[float]]
    y: Union[float, List[float]]
    z: Union[float, List[float]]
```

### Annotation tool
We provide the annotation tool for imitation learning dataset collection.
The user is presented by workspace heightmap, it has to select in the following gui grasp-point with angle.
The gripper is visible in the gui and its parameters can be adjusted to model real-life scenario.
Use `q` and `e` keys to change the grasping angle.
One can choose to additionally verify the human-annotated picks by the robot or always return success on those. 

![alt text](visualizations/labelling.png)

## Takeaways
* Imitation learning works for deformables achieving almost 90% success rate using less than 400 demonstrations
* For deformables even random baseline achieves reasonable grasp success rate
* Two-items picks are a problem, they have to be filtered through weight sensor

## Limitations and further steps
* We use only depth as input
  * using additionally rgb input should result in better GSR, but it may make model less robust to lighting conditions
* The pipeline is limited to top-down grasps
  * we can allow other gripper rotational poses for grasping
* The z-coord at which the grasp occur is determined by depth in (x, y) of the grasp
  * we can allow for the model to decide about the z-coordinate of the grasp attempt
* The initial two-finger gripper opening is set as fixed
  * we can allow for the model to decide about the initial opening value of the finger gripper
* All of the former should increase the robustness and GSR of the pipeline
  * gripper opening and z-coordinate of the grasp should be the easiest to implement, because we still will be limiting ourselves
to top-down grasps. Thus, the pipeline will not change much. Easiest solution would be to in addition to (x, y) affordance to output
dense maps with coresponding initial gripper opening and depth of the grasp as in https://arxiv.org/abs/1909.04810v4. 
* Sometimes the grasp results in grabing two cables, this should be solvable:
  * by providing more training data in imitation
  * by reading the weight sensors on the robot and re-grasping if needed
* Sometimes the framework detects grasp as not-successful, even if the cable is in the gripper,
because the success of the grasp is detected only by threshold on gripper-opening
  * again, reading the weight sensor as the grasping success-detecter will solve the thing
* Sometimes cable falls out during the place to bin move
  * again, this can be detected using weight sensor, then trying to regrasp 