# KinfuROS
**Authors:** Parker Ewen (pewen@umich.edu), Gitesh Sambhaji Gunjal (gitesh@umich.edu), Hao Chen (haochern@umich.edu), Anran Li (anranli@umich.edu), Yuzhen Chen (yuzhench@umich.edu), and Ram Vasudevan (ramv@umich.edu). 

## Dependencies
The package is built on Ubuntu 20.04 with ROS Noetic Distribution. Make sure the ROS Noetic is installed before downloading the dependencies.
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install ros-noetic-desktop
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Building

1. Create a catkin workspace.
```
mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/src
```

2. Download the package in src folder, make sure in main branch.
```
git clone https://github.com/anranli2003/KinfuROS
```
3. Build the package using either `catkin_make` or `catkin build` in catkin_ws folder.
```
catkin build
```

4. Source the workspace.
```
source devel/setup.bash
```
## Building the seg_preproc
1. Download the Depenedencies for k4a
   ```
   https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb
   https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.1_amd64.deb
   optionally: https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools/k4a-tools_1.4.1_amd64.deb
   ```
2. Download Tensor RT from Nvidia's website (Version 8.6.1 GA), assuming CUDA has already been installed
   ```
   https://developer.nvidia.com/nvidia-tensorrt-8x-download 
   ```
## Usage
Before running the package, make sure you build the package successfully and source the workspace.
1. Run the preproc package (
```
python3 ~/catkin_ws/src/seg_preproc/scripts/preproc.py
```
2 Launch the package using the following code.
```
roslaunch kinfu_semantic kinect_fusion_rviz.launch
```
3. Run the ROS bag containing pre-recorded
```
rosbag play {$bagname}.bag

