An overview of the opencv_contrib modules
-----------------------------------------

This list gives an overview of all modules available inside the contrib repository.
To turn off building one of these module repositories, set the names in bold below to <reponame>

```
$ cmake -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -D BUILD_opencv_<reponame>=OFF <opencv_source_directory>
```

- **kinfu**: RGB-Depth Processing module -- Linemod 3D object recognition; Fast surface normals and 3D plane finding. 3D visual odometry. 3d reconstruction using KinectFusion.
