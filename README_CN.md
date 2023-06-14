# LIO-Lite
一个可以用在无人机上的轻量级 LIO 系统：适配了Livox Mid-360雷达。  
* 整个项目是在[faster-lio](https://github.com/gaoxiang12/faster-lio.git)的基础上做适配  
* 增加了场景中的重定位  

## 分支
分出了基于优化的分支和基于滤波的分支。  
基于优化的分支在X86架构上测试表现良好，但是在英伟达NX上帧率只有5hz不到。  
基于滤波的算法在英伟达NX上表现不错，运行起来CPU总占用在30%~40%。除去一些基础的占用，总占用应该更低。

## Illustrate
c++ == 17  

## Dependence
1. ROS (melodic or noetic)
2. glog: ```sudo apt-get install libgoogle-glog-dev```
3. eigen: ```sudo apt-get install libeigen3-dev```
4. pcl: ```sudo apt-get install libpcl-dev```
5. yaml-cpp: ```sudo apt-get install libyaml-cpp-dev```

## Build
```
  git clone https://github.com/Liansheng-Wang/LIO-Lite.git  
  cd LIO-Lite  
  catkin_make  
```

## Run
```
  mkdir src/LIO-Lite/maps  src/LIO-Lite/logs
  source devel/setup.zsh
  roslaunch lio_lite mapping_360.launch  
```
在完成建图之后，可以使用下面的命令开启重定位模式：
```
  roslaunch lio_lite location_360.launch  
```
### note：
  重定位过程中如果不是在起点的时，需要在rviz中手动进行重定位

## Reference
* [FAST-LIO2](https://github.com/hku-mars/FAST_LIO.git)
* [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM.git)
* [faster-lio](https://github.com/gaoxiang12/faster-lio.git)