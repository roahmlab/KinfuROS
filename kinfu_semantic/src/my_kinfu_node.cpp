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
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/rgbdroahm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>  
#include <message_filters/sync_policies/approximate_time.h>


class KinectFusionNode {
private:
    ros::NodeHandle nh_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
    message_filters::Subscriber<sensor_msgs::Image> semantic_sub_;
    ros::Publisher visualization_pub_;
    image_transport::Publisher pub;
    Ptr<kinfuroahm::Params> params;
    Ptr<kinfuroahm::KinFu> kf;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_policy;
    message_filters::Synchronizer<sync_policy> sync;

public:
   
    KinectFusionNode() : nh_("~"), depth_sub_(nh_, "/preproc/depth", 1), semantic_sub_(nh_, "/preproc/segmentation", 1),
    sync(sync_policy(30), depth_sub_, semantic_sub_) {
        image_transport::ImageTransport it(nh_);
        pub = it.advertise("kinect_fusion_visualization", 1);
        sync.registerCallback(boost::bind(&KinectFusionNode::callback, this, _1, _2));

        int width = 1280;
        int height = 720;
        params = kinfuroahm::Params::defaultParams();
        initialize_kinfu_params(*params, width, height, 525, 525, width/2-0.5f, height/2-0.5f);
        kf = kinfuroahm::KinFu::create(params);
        
    }

    void callback(const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::ImageConstPtr& semantic_msg) {
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

        // Store the depth and semantic images
        cv::Mat depth = depth_cv_ptr->image.clone();
        cv::Mat semantic = semantic_cv_ptr->image.clone();

        if (!depth.empty() && !semantic.empty()) {
            // Update KinectFusion using kf_ object
            if (!kf->update(depth, semantic)) {
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

            //create the sensor message
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
