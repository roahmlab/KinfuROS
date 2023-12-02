#include "precomp.hpp"
#include "fast_icp.hpp"
#include "tsdf.hpp"
#include "kinfu_frame.hpp"

namespace cv {
namespace kinfuroahm {

Ptr<Params> Params::defaultParams()
{
    Params p; 

    p.frameSize = Size(1280, 720);

    float fx, fy, cx, cy;
    fx = fy = 525.f;
    cx = p.frameSize.width/2 - 0.5f;
    cy = p.frameSize.height/2 - 0.5f;
    p.intr = Matx33f(fx,  0, cx,
                      0, fy, cy,
                      0,  0,  1);

    // 5000 for the 16-bit PNG files
    // 1 for the 32-bit float images in the ROS bag files
    p.depthFactor = 5000;

    // sigma_depth is scaled by depthFactor when calling bilateral filter
    p.bilateral_sigma_depth = 0.04f;  //meter
    p.bilateral_sigma_spatial = 4.5; //pixels
    p.bilateral_kernel_size = 7;     //pixels

    p.icpAngleThresh = (float)(30. * CV_PI / 180.); // radians
    p.icpDistThresh = 0.1f; // meters

    p.icpIterations = {10, 5, 4};
    p.pyramidLevels = (int)p.icpIterations.size();

    p.tsdf_min_camera_movement = 0.001f; //meters, disabled

    p.volumeDims = Vec3i::all(512); //number of voxels

    float volSize = 4.f;
    p.voxelSize = volSize/512.f; //meters

    // default pose of volume cube
    // p.volumePose = Affine3f().translate(Vec3f(-volSize/2.f, -volSize/2.f, 0.5f)); //-----------> position of the cube
    p.volumePose = Affine3f().translate(Vec3f(0.f, 0.f, 0.f));

    p.tsdf_trunc_dist = 0.04f; //meters;
    p.tsdf_max_weight = 64;   //frames

    p.raycast_step_factor = 0.25f;  //in voxel sizes

    p.lightPose = Vec3f::all(0.f); //meters

    return makePtr<Params>(p);
}

Ptr<Params> Params::coarseParams()
{
    Ptr<Params> p = defaultParams();

    p->icpIterations = {5, 3, 2};
    p->pyramidLevels = (int)p->icpIterations.size();

    float volSize = 3.f;
    p->volumeDims = Vec3i::all(128); //number of voxels
    p->voxelSize  = volSize/128.f;

    p->raycast_step_factor = 0.75f;  //in voxel sizes

    return p;
}

// T should be Mat or UMat
template< typename T >
class KinFuImpl : public KinFu
{
public:
    KinFuImpl(const Params& _params);
    virtual ~KinFuImpl();

    const Params& getParams() const CV_OVERRIDE;

    void render(OutputArray image, const Matx44f& cameraPose) const CV_OVERRIDE;

    void getCloud(OutputArray points, OutputArray normals) const CV_OVERRIDE;
    void getPoints(OutputArray points) const CV_OVERRIDE;
    void getNormals(InputArray points, OutputArray normals) const CV_OVERRIDE;

    void reset() CV_OVERRIDE;

    const Affine3f getPose() const CV_OVERRIDE;


    bool update(InputArray depth, const Semantic& _semantic, const Affine3f& kinfuPose) override;

    bool updateT(const T& depth, const Semantic& _semantic, const Affine3f& kinfuPose);

private:
    Params params;

    cv::Ptr<ICP> icp;
    cv::Ptr<TSDFVolume> volume;
    
    int frameCounter;
    Affine3f pose;
    std::vector<T> pyrPoints;
    std::vector<T> pyrNormals;
    std::vector<T> pyrClasses;

};


template< typename T >
KinFuImpl<T>::KinFuImpl(const Params &_params) :
    params(_params),
    icp(makeICP(params.intr, params.icpIterations, params.icpAngleThresh, params.icpDistThresh)),
    volume(makeTSDFVolume(params.volumeDims, params.voxelSize, params.volumePose,
                          params.tsdf_trunc_dist, params.tsdf_max_weight,
                          params.raycast_step_factor)),
    pyrPoints(), pyrNormals(), pyrClasses()
{
    reset();
}

template< typename T >
void KinFuImpl<T>::reset()
{
    frameCounter = 0;
    pose = Affine3f::Identity();
    volume->reset();
}

template< typename T >
KinFuImpl<T>::~KinFuImpl()
{ }

template< typename T >
const Params& KinFuImpl<T>::getParams() const
{
    return params;
}

template< typename T >
const Affine3f KinFuImpl<T>::getPose() const
{
    return pose;
}


template<>
bool KinFuImpl<Mat>::update(InputArray _depth, const Semantic& _semantic, const Affine3f& kinfuPose)
{

    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

    Mat depth;

    if(_depth.isUMat())
    {
        _depth.copyTo(depth);
        return updateT(depth, _semantic, kinfuPose);
    }
    else
    {
        return updateT(_depth.getMat(), _semantic, kinfuPose);
    }
}


template<>
bool KinFuImpl<UMat>::update(InputArray _depth, const Semantic& _semantic, const Affine3f& kinfuPose)
{
//    printf("_depth.size(): (%d, %d) params.frameSize: (%d, %d)\n",
//           _depth.size().width, _depth.size().height,
//           params.frameSize.width, params.frameSize.height);
    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize); // TODO

    UMat depth;
    if(!_depth.isUMat())
    {
        _depth.copyTo(depth);
        return updateT(depth, _semantic, kinfuPose);
    }
    else
    {
        return updateT(_depth.getUMat(), _semantic, kinfuPose);
    }
}


template< typename T >
bool KinFuImpl<T>::updateT(const T& _depth, const Semantic& _semantic, const Affine3f& kinfuPose)
{
    CV_TRACE_FUNCTION();


    T depth;
    if(_depth.type() != DEPTH_TYPE)
        _depth.convertTo(depth, DEPTH_TYPE);
    else
        depth = _depth;


    // Get camera pose
    Matx44f cameraPose = pose.matrix;

    std::vector<T> newPoints, newNormals, newClasses;
    makeFrameFromDepth(depth, newPoints, newNormals, newClasses, params.intr,
                       params.pyramidLevels,
                       params.depthFactor,
                       params.bilateral_sigma_depth,
                       params.bilateral_sigma_spatial,
                       params.bilateral_kernel_size);

    if(frameCounter == 0)
    {
        // use depth instead of distance
        volume->integrate(depth,_semantic, params.depthFactor, pose, params.intr);

        pyrPoints  = newPoints;
        pyrNormals = newNormals;
        pyrClasses = newClasses;
    }
    else
    {
        pose = kinfuPose;

//        float rnorm = (float)cv::norm(affine.rvec());
//        float tnorm = (float)cv::norm(affine.translation());
//
//
//        // We do not integrate volume if camera does not move
//        if((rnorm + tnorm)/2 >= params.tsdf_min_camera_movement)
//        {
//            printf("camera does not move\n");
//            // use depth instead of distance
//            volume->integrate(depth, _semantic, params.depthFactor, pose, params.intr);
//        }

        T& points  = pyrPoints [0];
        T& normals = pyrNormals[0];
        T& VoxMat = pyrClasses[0];
        volume->raycast(pose, params.intr, params.frameSize, points, normals, VoxMat);

        // build a pyramid of points and normals
        buildPyramidPointsNormals(points, normals, pyrPoints, pyrNormals,
                                  params.pyramidLevels);
    }

    frameCounter++;
    return true;
}


template< typename T >
void KinFuImpl<T>::render(OutputArray image, const Matx44f& _cameraPose) const
{ 
    CV_TRACE_FUNCTION();

    Affine3f cameraPose(_cameraPose);
 
    const Affine3f id = Affine3f::Identity();
    if((cameraPose.rotation() == pose.rotation() && cameraPose.translation() == pose.translation()) ||
       (cameraPose.rotation() == id.rotation()   && cameraPose.translation() == id.translation()))
    {
        printf("That happened1");
        T points, normals, voxelClass;
        volume->raycast(cameraPose, params.intr, params.frameSize, points, normals, voxelClass);
        renderPointsNormals(pyrPoints[0], pyrNormals[0], pyrClasses[0], image, params.lightPose);
    }
    else
    {   
        printf("This happened");
        T points, normals, voxelClass;
        volume->raycast(cameraPose, params.intr, params.frameSize, points, normals, voxelClass);
        renderPointsNormals(points, normals, voxelClass, image, params.lightPose); 
    }
}


template< typename T >
void KinFuImpl<T>::getCloud(OutputArray p, OutputArray n) const
{
    printf("\n 1 \n");
    volume->fetchPointsNormals(p, n);
}


template< typename T >
void KinFuImpl<T>::getPoints(OutputArray points) const
{
    volume->fetchPointsNormals(points, noArray());
}


template< typename T >
void KinFuImpl<T>::getNormals(InputArray points, OutputArray normals) const
{
    volume->fetchNormals(points, normals);
}


#ifdef OPENCV_ENABLE_NONFREE

Ptr<KinFu> KinFu::create(const Ptr<Params>& params)
{
#ifdef HAVE_OPENCL
    if(cv::ocl::useOpenCL())
        return makePtr< KinFuImpl<UMat> >(*params);
#endif
    return makePtr< KinFuImpl<Mat> >(*params);
}

#else
Ptr<KinFu> KinFu::create(const Ptr<Params>& /*params*/)
{
    CV_Error(Error::StsNotImplemented,
             "This algorithm is patented and is excluded in this configuration; "
             "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library");
}
#endif

KinFu::~KinFu() {}

} // namespace kinfuroahm
} // namespace cv
