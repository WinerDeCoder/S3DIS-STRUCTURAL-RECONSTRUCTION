# S3DIS-Structural-Completion
3D Point Cloud Structural Complete for S3DIS dataset ( make synthetic from real )

## What is S3DIS ?
The S3DIS dataset stands for Stanford Large-Scale 3D Indoor Scenes. It's a publicly available dataset widely used for research and development in the field of 3D scene understanding. [Detail information about S3DIS](http://buildingparser.stanford.edu/dataset.html)

<img src="http://buildingparser.stanford.edu/images/sampleData.png" width="800" alt="Screenshot of My Project">


### Purpose:

The S3DIS dataset provides a large collection of real-world indoor scenes captured using LiDAR scanners. These scans offer rich 3D point cloud data, allowing researchers to develop and test algorithms for various tasks related to indoor spaces, such as:

* **Semantic Segmentation:** Classifying each point in the point cloud according to its semantic category (e.g., wall, floor, ceiling, chair, table).
* **Object Detection and Recognition:** Identifying and locating objects within the scene.
* **3D Scene Completion and Reconstruction:** Creating a complete 3D model of the scene based on the point cloud data.

### Content:
The S3DIS dataset consists of:

* **6 large-scale indoor areas:** These areas represent diverse indoor environments, including offices, laboratories, and hallways.
* **271 rooms:** Each area is further divided into smaller rooms, providing a variety of spatial configurations for analysis.
* **Point cloud data:** LiDAR scans capture the 3D geometry of the scenes, with each point representing a location in space along with its intensity value (reflectance).
* **Semantic annotations:** Each point in the point cloud is labeled with a semantic category (e.g., ceiling, floor, window) for ground truth data in training and evaluating segmentation models.

## The Purpose of this Project
### Problem Statement
We focus on 3D Point Cloud Structure of S3DIS such as Walls, Ceiling, Floor, Column. Because S3DIS is captured from real life, the result is affected by many natural elements. So the point clouds of these planar are in noisy shape. As some pictures follow:
### WALL:
<p align="center">
<img src="https://github.com/WinerDeCoder/S3DIS-Structural-Completion/assets/136697023/7f6594a1-3e1d-42da-9553-3837b9524529" width="500" alt="Screenshot of My Project">
</p>
### CEILING and FLOOR:

<p align="center">
  <a href="https://getbootstrap.com" target="_blank" rel="noreferrer">
    <img src="https://github.com/WinerDeCoder/S3DIS-Structural-Completion/assets/136697023/973b5fd6-93b5-4aff-8647-4019d5045961" alt="bootstrap" width="400" height="400" style="padding-left:20px;padding-right:20px" />
  </a>
  <a href="#">  <img src="https://github.com/WinerDeCoder/S3DIS-Structural-Completion/assets/136697023/4fd67467-5ba7-4a9b-8f98-390897b287ba" alt="cplusplus" width="400" height="400" margin-right= "100px" ; />
  </a>
</p>
  
### COLUMN:
<p align="center">
<img src="https://github.com/WinerDeCoder/S3DIS-Structural-Completion/assets/136697023/8eec1b5f-5c9f-4ba0-8518-605a2dd8d264" width="500" alt="Screenshot of My Project">
</p>

### Solution
Now we will reconstruct complete clean, smooth 3D Point Clouds of them. The method we use is very simple. Only need to find the min, max of 3 coordinate (xyz) then use linspace to create new grid of points. Detail code are shown in Python file above, take times to read them if you want.

* Some example:

<p align="center">
<img src="https://github.com/WinerDeCoder/S3DIS-Structural-Completion/assets/136697023/9e79088b-9c67-4e69-a512-4126e24eaae1" width="700" alt="Screenshot of My Project" >

<img src="https://github.com/WinerDeCoder/S3DIS-Structural-Completion/assets/136697023/e1f1605b-6970-4d85-8f4f-86900bf39370" width="700" alt="Screenshot of My Project" >

<img src="https://github.com/WinerDeCoder/S3DIS-Structural-Completion/assets/136697023/4b72a536-9bb1-4067-9f76-34efa7af0311" width="700" alt="Screenshot of My Project" >

</p>

* Full Room

<p align="center">
<img src="https://github.com/WinerDeCoder/S3DIS-Structural-Completion/assets/136697023/51c32ad5-1a78-4759-8d7e-ab91727928c5" width="700" alt="Screenshot of My Project" >

<img src="https://github.com/WinerDeCoder/S3DIS-Structural-Completion/assets/136697023/2a0fa382-9b87-412d-bbf7-fe0e2d5a5c76" width="700" alt="Screenshot of My Project" >

<img src="https://github.com/WinerDeCoder/S3DIS-Structural-Completion/assets/136697023/01fdac8c-d826-4bf2-87f1-cc7f54859f7c" width="700" alt="Screenshot of My Project" >

</p>

## Acknowledgement
I want to thank you Pham Huy Thien Phuc for training resources and effort in idea and Relationship, Dr. Tuan Dang for an amazing opportunity working for his Lab











