#include "Eigen.h"
#include <iostream>
#include <fstream>

class PointCloud {
public:

	// 2D keypoints in the image
    std::vector<cv::KeyPoint> points2d;

    // Matrices
    Matrix4f cameraExtrinsics;
    Matrix3f cameraIntrinsics;

    // MAP: KEY => INDEX 2D KEYPOINT CURRENT FRAME | VALUE => INDEX 2D KEYPOINTS NEXT FRAME
    std::map<int, int> matching_index_with_next_frame;

    // MAP: KEY => INDEX 2D KEYPOINT CURRENT FRAME | VALUE => INDEX 2D KEYPOINTS PREV FRAME
    std::map<int, int> matching_index_with_prev_frame;
    
    // MAP: KEY => INDEX 2D KEYPOINT CURRENT FRAME | VALUE => INDEX 3D GLOBAL ARRAY
    std::map<int, int> global_3D_indices;

    // Constructor
    PointCloud(const Matrix3f& intrinsicMatrix, const Matrix4f& extrinsicMatrix){
        cameraIntrinsics = intrinsicMatrix;
        cameraExtrinsics = extrinsicMatrix;
    }

    //Getters and setters
    std::map<int, int> getGlobal3Dindices(){
        return global_3D_indices;
    }

    void appendGlobal3Dindices(const int key, const int value){
        global_3D_indices[key] = value;
    }


    void setIndexPrevMatchesFrames(const std::map<int, int>& matches){
        matching_index_with_prev_frame = matches;
    }

    std::map<int, int> getIndexPrevMatchesFrames(){
        return matching_index_with_prev_frame;
    }


    void setIndexMatchesFrames(const std::map<int, int>& matches){
        matching_index_with_next_frame = matches;
    }

    std::map<int, int> getIndexMatchesFrames(){
        return matching_index_with_next_frame;
    }


    Matrix4f getCameraExtrinsics() const{
        return cameraExtrinsics;
    }

    void setCameraExtrinsics(const Matrix4f& extrinsicMatrix){
        cameraExtrinsics = extrinsicMatrix;
    }


    Matrix3f getCameraIntrinsics() const{
        return cameraIntrinsics;
    }

    void setCameraIntrinsics(const Matrix3f& intrinsicMatrix){
        cameraIntrinsics = intrinsicMatrix;
    }


    std::vector<cv::KeyPoint> getPoints2d() const{
        return points2d;
    }

    void setPoints2d(const std::vector<cv::KeyPoint> points2dim){
        points2d = points2dim;
    }


    // Project a single 2D point into 3D
    Vector3f point2dToPoint3d(Vector2f point2d, float depth){

        float fovX = cameraIntrinsics(0, 0);
        float fovY = cameraIntrinsics(1, 1);
        float cX = cameraIntrinsics(0, 2);
        float cY = cameraIntrinsics(1, 2);

        Matrix4f cameraExtrinsicsInv = cameraExtrinsics.inverse();
        Matrix3f rotationInv = cameraExtrinsicsInv.block(0, 0, 3, 3);
        Vector3f translationInv = cameraExtrinsicsInv.block(0, 3, 3, 1);

        if (depth == MINF) {
            printf("It should not enter here.\n");
            return Vector3f(MINF, MINF, MINF);
        }

        float x = ((float) point2d.x() - cX) / fovX;
        float y = ((float) point2d.y() - cY) / fovY;

        Vector4f backprojected = Vector4f(depth * x, depth * y, depth, 1);
        Vector4f worldSpace = cameraExtrinsicsInv * backprojected;

        return Vector3f(worldSpace[0], worldSpace[1], worldSpace[2]);
    }
    
};