#include <ros/ros.h>

#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>

#include "tf/transform_datatypes.h"
#include "tf/LinearMath/Matrix3x3.h"

#include "mpc_controller_bindings.hpp"

struct VehicleStatus
{
    std_msgs::Header header;
    geometry_msgs::Pose pose;
    double roll;
    double pitch;
    double yaw;
};

VehicleStatus vehicle_status_;

double x_reference[3] = {0,0,0};

void CommandPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg)
{
    x_reference[0] = msg->pose.position.x;
    x_reference[1] = msg->pose.position.y;
    x_reference[2] = msg->pose.orientation.w;
//     ROS_INFO("x_r[0]: %f, x_r[1]: %f, x_r[2]: %f", x_reference[0], x_reference[1], x_reference[2]);
}


void odometryCallback(const nav_msgs::Odometry::ConstPtr &msg) {
    ROS_INFO_ONCE("Control interface got first odometry message.");
    vehicle_status_.header = msg->header;
    vehicle_status_.pose = msg->pose.pose;
    // Stamp odometry upon reception to be robust against timestamps "in the future".
    vehicle_status_.header.stamp = ros::Time::now();
    // Convert the orientation quaternion to RPY and save it.
    tf::Quaternion quat(vehicle_status_.pose.orientation.x, vehicle_status_.pose.orientation.y, vehicle_status_.pose.orientation.z, vehicle_status_.pose.orientation.w);
    tf::Matrix3x3(quat).getRPY(vehicle_status_.roll, vehicle_status_.pitch, vehicle_status_.yaw);
}
    

int main(int argc, char** argv)
{
    int i;
    double ctrl_period_;
    double lin_vel_cmd = 0, ang_vel_cmd = 0;
    std::string out_twist_, in_odometry_; //, out_vehicle_cmd, in_vehicle_status, in_waypoints, in_selfpose;


    ros::init(argc, argv, "open_mpc_controller");
    ros::NodeHandle nh, private_nh_("~");
    
    ROS_INFO("Initializing %s node.", ros::this_node::getName().c_str());
    
    // Setup OpEn solver

    /* parameters             */
    double p[MPC_CONTROLLER_NUM_PARAMETERS] = {2.0, 10.0};

    /* initial guess          */
    double u[MPC_CONTROLLER_NUM_DECISION_VARIABLES] = {0};

    /* initial penalty        */
    double init_penalty = 15.0;

    /* obtain cache           */
    mpc_controllerCache *cache = mpc_controller_new();
    
    private_nh_.param("control_period", ctrl_period_, double(0.1));
    private_nh_.param("out_twist_name", out_twist_, std::string("twist_raw"));
    private_nh_.param("in_odom_name", in_odometry_, std::string("odometry"));
//     timer_control_ = nh_.createTimer(ros::Duration(ctrl_period_), &NonLinearModelPredictiveControllerAcados::timerCallback, this);

    ros::Subscriber pos_sub = nh.subscribe("/odometry/filtered", 1, odometryCallback);
    ros::Subscriber command_trajectory_subscriber = nh.subscribe("command/pose", 1, CommandPoseCallback);
    ros::Publisher pub_twist_cmd_ = nh.advertise<geometry_msgs::Twist>(out_twist_, 1);

//   ros::spin();
    ros::Rate rate((int)(1.0/ctrl_period_));

    while(ros::ok()) {
        ROS_INFO("Position (global variables)-> x: [%f], y: [%f], z: [%f]", vehicle_status_.pose.position.x, vehicle_status_.pose.position.y, vehicle_status_.pose.position.z);
        
        p[0] = vehicle_status_.pose.position.x;
        p[1] = vehicle_status_.pose.position.y;
        p[2] = vehicle_status_.yaw;
        p[3] = x_reference[0];
        p[4] = x_reference[1];
        p[5] = x_reference[2];
        
        /* solve                  */
        mpc_controllerSolverStatus status = mpc_controller_solve(cache, u, p, 0, &init_penalty);
        
        lin_vel_cmd = u[0];
        ang_vel_cmd = u[1];
    
        geometry_msgs::Twist twist;
        twist.linear.x = lin_vel_cmd;
        twist.linear.y = 0.0;
        twist.linear.z = 0.0;
        twist.angular.x = 0.0;
        twist.angular.y = 0.0;
        twist.angular.z = ang_vel_cmd;
        pub_twist_cmd_.publish(twist);
        
        ROS_INFO("Solve time: %f ms. I will send %f %f \n", (double)status.solve_time_ns / 1000000.0, lin_vel_cmd, ang_vel_cmd);
        

        ros::spinOnce();
        rate.sleep();
    }
    
    /* free memory */
    mpc_controller_free(cache);
    
    return 0;
}
