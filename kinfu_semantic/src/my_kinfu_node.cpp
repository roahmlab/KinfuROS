#include <ros/ros.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <k4a/k4a.h>
#include <math.h>
#include "my_kinfu_node.h"
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/rgbdroahm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>  
#include <message_filters/sync_policies/approximate_time.h>


cv::Matx44f poseToMatrix(const geometry_msgs::Point& position, const geometry_msgs::Quaternion& orientation)
{
//    TODO: potential accuracy issue
//    Position: x: -0.146726, y: 0.215163, z: 0.076227
//    Orientation: x: 0.002670, y: 0.035435, z: -0.063024, w: 0.997379
//                      0.989470 0.126767 0.069855 -0.148232
//                      -0.126363 0.991932 -0.010202 0.216564
//                      -0.070585 0.001267 0.997505 0.075367
//                      0.000000 0.000000 0.000000 1.000000

    Eigen::Quaterniond quat(orientation.w, orientation.x, orientation.y, orientation.z);

    Eigen::Translation3d trans(position.x, position.y, position.z);

    Eigen::Matrix4f transEigen = (trans * quat).matrix().cast<float>();

    cv::Matx44f transOpenCV;
    cv::eigen2cv(transEigen, transOpenCV);

    return transOpenCV;
}

void printMat(const cv::Matx44f& mat)
{
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++)
            printf("%f ", mat(i, j));
        printf("\n");
    }
}

class KinectFusionNode {
private:
    ros::NodeHandle nh_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
    message_filters::Subscriber<sensor_msgs::Image> semantic_sub_;
    message_filters::Subscriber<geometry_msgs::PoseWithCovarianceStamped> pose_sub_;
    ros::Publisher visualization_pub_;
    image_transport::Publisher pub;
    Ptr<kinfuroahm::Params> params;
    Ptr<kinfuroahm::KinFu> kf;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                            sensor_msgs::Image,
                                                            geometry_msgs::PoseWithCovarianceStamped> sync_policy;
//    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_policy;

    message_filters::Synchronizer<sync_policy> sync;

public:

    KinectFusionNode() : nh_("~"),
                         depth_sub_(nh_, "/preproc/depth", 1),
                         semantic_sub_(nh_, "/preproc/segmentation", 1),
                         pose_sub_(nh_, "/rtabmap/rtabmap/localization_pose", 1),
//                         sync(sync_policy(30), depth_sub_, semantic_sub_) {
                         sync(sync_policy(30), depth_sub_, semantic_sub_, pose_sub_)
    {
        image_transport::ImageTransport it(nh_);
        pub = it.advertise("kinect_fusion_visualization", 1);
        sync.registerCallback(boost::bind(&KinectFusionNode::callback, this, _1, _2, _3));
//        sync.registerCallback(boost::bind(&KinectFusionNode::callback, this, _1, _2));

        int width = 1280;
        int height = 720;
        params = kinfuroahm::Params::defaultParams();
        initialize_kinfu_params(*params, width, height, 525, 525, width/2-0.5f, height/2-0.5f);
        kf = kinfuroahm::KinFu::create(params);

    }

    void callback(const sensor_msgs::ImageConstPtr& depth_msg,
                  const sensor_msgs::ImageConstPtr& semantic_msg,
                  const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg) {
//    void callback(const sensor_msgs::ImageConstPtr& depth_msg,
//                  const sensor_msgs::ImageConstPtr& semantic_msg) {
        // Convert ROS Image to cv::Mat
        ROS_INFO("received message");
        cv_bridge::CvImagePtr depth_cv_ptr, semantic_cv_ptr;
        try {
            depth_cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
            semantic_cv_ptr = cv_bridge::toCvCopy(semantic_msg, sensor_msgs::image_encodings::TYPE_8UC1);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        geometry_msgs::Point p = pose_msg->pose.pose.position;
        geometry_msgs::Quaternion q = pose_msg->pose.pose.orientation;
        cv::Matx44f cameraPose = poseToMatrix(p, q);
        printMat(cameraPose);

        ROS_INFO("Position: x: %f, y: %f, z: %f", p.x, p.y, p.z);
        ROS_INFO("Orientation: x: %f, y: %f, z: %f, w: %f", q.x, q.y, q.z, q.w);

        // Store the depth and semantic images
        cv::Mat depth = depth_cv_ptr->image.clone();
        cv::Mat semantic = semantic_cv_ptr->image.clone();

        if (!depth.empty() && !semantic.empty()) {
            // Update KinectFusion using kf_ object
            if (!kf->update(depth, semantic, cameraPose)) {
                printf("Reset KinectFusion\n");
                kf->reset();
            }
            // Retrieve rendered TSDF
            UMat tsdfRender;
            kf->render(tsdfRender);

            UMat points;
            UMat normals;

            // Convert the UMat to a cv::Mat
            cv::Mat tsdfRenderMat;
            tsdfRender.copyTo(tsdfRenderMat);

            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgba8", tsdfRenderMat).toImageMsg();
            pub.publish(msg);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "kinect_fusion_node");
    // ros::NodeHandle nh;
    KinectFusionNode node;
    ROS_INFO("Node initialized");
    
    ros::spin();
    return 0;
}
