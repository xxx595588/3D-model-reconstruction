# 3D-model-reconstruction

## Table of contents
* [Overview](#overview)
* [Data](#data)
* [Algorithms](#algorithms)
* [Results](#results)
* [Assessment and Evaluation](#assessment-and-evaluation)

## Overview
&emsp;&emsp;In this project, we built the python codes to reconstruct the 3D model by feeding different sets of object images in terms of angle. Briefly, we will obtain the accurate intrinsic parameters of the scanned cameras by calibrate.py which took a collection of images (background) taken from different angles in the calib_jpg_u folder. Then, we decode a 10-bit gray code pattern with the given difference threshold to obtain view code and masks by calling decode() which was encapsulated in reconstruct(). After completing the reconstruct process, we called mesh() to do the bounding box pruning, triangle pruning, and mesh smoothing for each grab. The last step is to use meshLab to combine all mesh results together.

Below is the diagram of how the project work:
<p align="center">
  <img width="684" alt="Screen Shot 2023-04-08 at 9 48 51 AM" src="https://github.com/xxx595588/3D-model-reconstruction/assets/61955371/63c6188c-b92d-407f-9ed3-06e50d256dfd">
</p>

## Data

<p align="center">
  <img  alt="Screen Shot 2023-04-08 at 9 48 51 AM" src="https://github.com/xxx595588/3D-model-reconstruction/assets/61955371/2ac2b3f6-ac49-410a-abed-394b9b4c4a8e">
</p>
&emsp;&emsp;Within the scope of this project, we have employed a teapot as our designated target object. Specifically, we have organized the teapot-related data into a series of seven folders, each corresponding to distinct shot angles. These folders contain collections of both object images and structured light images. The purpose of using object images is that it helps us to recognize and compute the real mask of the object at a certain shot angle, since we might encounter the issues that the color on the object is the same as the background in some structured light images. Furthermore, the main purpose of using structured light images is that we can simply convert it from binary into decimal.


## Algorithms
In this section, we will illustrate each module respectively according to our diagram in the Project Overview section.

- Cameras Parameters Acquisition
  
  In order to acquire the accurate intrinsic parameters of the scanned cameras, we will run calibrate.py which takes images (taken from different angles) as input from the calib_jpg_u folder. After finishing running calibrate.py, we will have a file called calibration.pickle which represents the intrinsic parameters of the scanned cameras with few parameters (focal length, center coordinate, rotation matrix, and translation) which will be useful for reconstructing the object in the following module.

<p align="center">
  <img alt="Screenshot 2023-06-14 190900" src="https://github.com/xxx595588/3D-model-reconstruction/assets/61955371/49a13d15-12c9-4eea-9a97-a58ae33e1df4">
</p>

- Image Decode
  
  In the decode function, we will decode a 10 bit gray code pattern with the given difference threshold, a mask of the given structured light images, and a mask of the given object image.The main reason we need the object mask is because in some structured light images, the surface of the object might have the same value as the background (both are covered by the shadow). Consequently, by utilizing an object mask, we gain the advantage of effortlessly isolating the genuine form and contours of the object, enabling us to discern its true shape amidst the confounding visual interplay. Below is how we complete the job:
  1. Read in 2 images (color_C0_00/01). One is the pure background, the other one is the object placed in front of the background, taking the difference of 2 arrays.
  2. After taking the difference, we will compare with the threshold for setting up our object mask. For all values that are greater than the threshold, we set it to 0. Otherwise, set it to 1.
  <p align="center">
     <img alt="Screenshot 2023-06-14 190900" src="https://github.com/xxx595588/3D-model-reconstruction/assets/61955371/3ab4942d-00d7-4430-9c7a-6cbb42b1afc5">
  </p>
  
- Reconstruct
  
  In the process of executing the reconstruct function, our aim is to perform two crucial operations, namely triangulation and computation of color values. These operations are accomplished by invoking the triangulate() and values_helper() subroutines respectively. The primary objective of the triangulate function is to transform the set of provided points (pts2L & pts2R) captured from two distinct camera shots captured at divergent angles into a unified representation in the form of global 3D coordinates. Within the triangulate function, we employ the np.linalg.lstsq() method to yield the optimal solution with the least amount of error, effectively solving our linear equation. In the values_helper(), we read in 2 object images and extract the corresponding points in given images from pts2L & pts2R, then we will take the average between them.
  <p align="center">
    <img width="500" alt="Screenshot 2023-06-14 192031" src="https://github.com/xxx595588/3D-model-reconstruction/assets/61955371/5f864a65-f01f-4072-9781-ed4458437fb1">
  </p>

- Mesh
  
  Our mesh function will comprise a set of three fundamental techniques, namely Bounding Box Pruning, Triangle Pruning, and Mesh Smoothing. By skillfully integrating these three approaches, we can confidently ascertain the elimination of redundant data points, thereby ensuring the maintenance of superior mesh quality throughout the process.
  <p align="center">
    <img alt="Screenshot 2023-06-14 192031" src="https://github.com/xxx595588/3D-model-reconstruction/assets/61955371/a2aa461a-60dd-4f34-8887-3c8bebe3d9a4">
  </p>

  1. Bounding Box Pruning
     
      In the Bounding Box Pruning function, we will pass all of the points (pts2L & pts2R & pts3), bounding box limit, and the object color values array. Then it will call bbp_con() helper function to determine which points to drop in bbp_drop. After deleting those points outside of the box, we will return the trimmed collection of the points (pts2L & pts2R & pts3), bounding box limit, and the object color values array.
      <p align="center">
        <img width="500" alt="Screenshot 2023-06-14 192031" src="https://github.com/xxx595588/3D-model-reconstruction/assets/61955371/3bd92111-a6de-41eb-a19c-755ecc8ce492">
      </p>

  2. Triangle Pruning

      In the Triangle Pruning function,this function will take Delaunay object, pts3, object color values array, and the triangle pruning threshold as parameters. Then it will call the tp_con() helper function to determine what points to be removed based on the triangle pruning threshold. After deleting triangles from the surface mesh that include edges that are greater than triangle pruning threshold, it will return the trimmed collection of pts3, tri.simplices, and the object color values array.
       <p align="center">
          <img width="500" alt="Screenshot 2023-06-14 192031" src="https://github.com/xxx595588/3D-model-reconstruction/assets/61955371/abbce80a-6844-4f23-8a68-9f35eb4b9da1">
        </p>

  4. Mesh Smoothing

      The last step is to apply the mesh smoothing function, mesh_smooth() will take Delaunay object, step(how many iterations of smoothing will be), and pts3. This function just simply takes the average of the neighborhood points with the current point.
     <p align="center">
        <img width="500" alt="Screenshot 2023-06-14 192031" src="https://github.com/xxx595588/3D-model-reconstruction/assets/61955371/772fb72b-3105-4798-ad75-cba74b29437c">
      </p>

- Combine with MeshLab
  
  The subsequent step in this project involves the consolidation of all the individual meshes. To accomplish this, we will employ the use of a software tool called MeshLab. Our approach will entail initially focusing on aligning the meshes that exhibit overlapping areas. This alignment process will employ a technique known as point based gluing, which offers the advantage of achieving a more precise and accurate mesh alignment. However, in cases where no overlapping areas exist, manual alignment becomes the sole option available, albeit with the potential drawback of introducing some imperfections or inconsistencies along the mesh edges.

## Results
<p align="center">
  <img width="500" alt="Screenshot 2023-06-14 192031" src="https://github.com/xxx595588/3D-model-reconstruction/assets/61955371/dc7c22fa-762e-4738-8bdb-4c33de7fdb41">
</p>

## Assessment and Evaluation
&emsp;&emsp;In the context of this particular project, the aspect that presents the greatest challenge is the Mesh component. This is due to the fact that it involves a series of three distinct procedures that must be executed in order to effectively eliminate superfluous points. These procedures encompass Bounding Box Pruning, Triangle Pruning, and Mesh Smoothing. For instance, when dealing with Bounding Box Pruning, it becomes necessary to determine the appropriate range along the x, y, and z axis, with the aim of ensuring that crucial points are not mistakenly deleted. Similarly, Triangle Pruning demands meticulous experimentation to identify the optimal threshold for selecting points, thereby necessitating a significant number of trial runs to ascertain the most suitable parameters for generating a satisfactory mesh output. 

&emsp;&emsp;Based on the results obtained thus far, it has become evident that certain meshes exhibit strong compatibility with one another, mainly due to the presence of numerous overlapping regions. Notably, meshes such as grab0, grab1, grab3, and grab4 have displayed a mesh error rate that typically falls below 1%. Conversely, it is readily discernible that meshes lacking or possessing minimal overlapping areas, such as grab2 and grab6, as well as areas obscured by shadows, tend to exhibit noticeable cutting edges. To address this issue effectively, one straightforward solution involves capturing additional images that possess a greater degree of overlap with those that currently lack it. Through this approach, it becomes possible to minimize the occurrence of mesh errors and enhance the overall quality of the output.

&emsp;&emsp;The project also addresses the issue of noise, specifically in filtering out distant noise from the object by utilizing Bounding Box Pruning. However, certain noise that is in close proximity to the object cannot be eliminated entirely. To mitigate this, mesh smoothing techniques are employed three times for each grab in this project to minimize the distance of the noise from the object. Nonetheless, this approach has a drawback as it leads to a loss of surface detail due to excessive averaging.






