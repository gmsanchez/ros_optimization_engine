# ros_optimization_engine

ROS meets Optimization Engine and they drive a Clearpath Robotics Husky.

You can send the waypoint to the `/test/command/pose` topic and the MPC controller will try to get the vehicle there.

You must have a working catkin workspace with Clearpath Robotics Husky package. Then you can issue

```
roslaunch husky_gazebo husky_empty_world.launch
roslaunch open_mpc_controller open_mpc_controller.launch
```

Thanks to:

- [Optimization Engine](https://alphaville.github.io/optimization-engine/)
- [CasADi](https://web.casadi.org/)
- [Clearpath Robotics Husky](https://github.com/husky/husky)
