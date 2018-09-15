#include <iostream>
#include <fstream>
#include <ctime>
#include <set>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "Eigen.h"
#include "PointCloud.h"
#include "Optimization.h"

using namespace std;
using namespace cv;

#define FRAMES_TO_OPTIMIZE 90
#define SKIP_FRAMES 5
#define POINT_CONSECUTIVE_FRAMES 5
#define WINDOW_SIZE 10

std::string DATA_PATH = PROJECT_DIR + std::string("/data/rgbd_dataset_freiburg3_long_office_household/");

// Prototype functions
void triangulationWithLastFrame(std::vector<PointCloud>& frames,std::vector<Vector3f>& global_3D_points,int last_frame_ind);
int consecutiveFrames(std::vector<PointCloud>& , int , int , int );
void generateOffFile(std::string filename, std::vector<Vector3f> points3d, std::vector<cv::Vec3b> colorValues, std::vector<PointCloud> pointClouds, int breakingPoints);
std::vector<Vector3f> performTriangulation(std::vector<Vector2f> points2dFrame1, std::vector<Vector2f> points2dFrame2, Matrix4f extrinsicsFrame1, Matrix4f extrinsicsFrame2, Matrix3f cameraIntrinsics);
void get_data(std::string, std::vector<cv::Mat>& depth_images, std::vector<cv::Mat>& rgb_images, std::vector<Matrix4f>& transformationMatrices);
Matrix4f getExtrinsicsFromQuaternion(std::vector<string>);
void split(const std::string &s, char delim, std::vector<std::string> &elems);


int main(int argc, char *argv[]){

	int i, j;

    std::cout << "PHASE 0: Get needed data" << std::endl;    

    std::vector<cv::Mat> depthImages, rgbImages;
    std::vector<Matrix4f> transformationMatrices;

    get_data("final_mapping.txt", depthImages, rgbImages, transformationMatrices);

    Matrix3f intrinsicMatrix;
    intrinsicMatrix <<  525.0f, 0.0f, 319.5f,
                        0.0f, 525.0f, 239.5f,
                        0.0f, 0.0f, 1.0f;

    std::cout << "PHASE 1: Finding, matching, discarding outlier keypoints" << std::endl;

    // Detect keypoints
    std::vector<std::vector<cv::KeyPoint>> keypointsAllImgs;
    cv::Ptr<cv::ORB> detectorORB = cv::ORB::create(2500);

    // Descriptors for keypoints
    std::vector<cv::Mat> descriptorsAllImgs;
    cv::Ptr<cv::BRISK> detectorBRISK = cv::BRISK::create();

    // Matching descriptors, we use NORM_HAMMING because the descriptors are BINARY and so it is faster with this norm
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    int nearest_neighbors = 2;
    const float ratio = 0.60;    //parameter to be tuned

    std::vector<std::vector<std::vector<cv::DMatch>>> allMatches;

    // Detect keypoints and calculate descriptors
    for (i=0; i<rgbImages.size(); i++){
        std::vector<cv::KeyPoint> keypointsImg;
        cv::Mat descriptorsImg;

        detectorORB->detect(rgbImages[i], keypointsImg);
        detectorBRISK->compute(rgbImages[i], keypointsImg, descriptorsImg);

        keypointsAllImgs.push_back(keypointsImg);
        descriptorsAllImgs.push_back(descriptorsImg);

        if (i > 0){
            // Match with kNN algorithm between pair of frames
            std::vector<std::vector<cv::DMatch>> matches;
            matcher.knnMatch(descriptorsAllImgs[i-1], descriptorsAllImgs[i], matches, nearest_neighbors);
            allMatches.push_back(matches);
        }
    }

    // Array of frames with corresponding information
    std::vector<PointCloud> pointClouds;

    // Initialize all the frames
    for(i=0; i < rgbImages.size(); i++){

    	Matrix4f extrinsicMatrix;

    	// We take ground truth poses from the first frame only (for testing purpose, identity is also ok)
		if (i == 0)
			extrinsicMatrix = transformationMatrices[i];
		else
			extrinsicMatrix = Matrix4f::Identity();

		// Create a new point cloud of the current frame
		PointCloud pointCloud = PointCloud(intrinsicMatrix, extrinsicMatrix);
		pointClouds.push_back(pointCloud);
    }

	// Remove outliers
	for (i = 0; i<(rgbImages.size() - 2); i++) {

		// Matches between first and third frame to check if keypoint is not noise
		std::vector<std::vector<cv::DMatch>> matchesFirstThirdFrame;
		matcher.knnMatch(descriptorsAllImgs[i], descriptorsAllImgs[i + 2], matchesFirstThirdFrame, nearest_neighbors);

		// Vector of index match between current and nextframe
		std::map<int, int> good_matches, good_matches_prev;

		for (j = 0; j<allMatches[i].size(); j++) {

			// Discard possible outliers, between first and second frame, if the 2 Nearest Neighbors are too close each other
			if (allMatches[i][j][0].distance < ratio * allMatches[i][j][1].distance) {
				// Discard possible outliers, between first and second frame, if the 2 Nearest Neighbors are too close each other
				if (allMatches[i + 1][allMatches[i][j][0].trainIdx][0].distance < ratio * allMatches[i + 1][allMatches[i][j][0].trainIdx][1].distance) {
					// Find the same keypoint both in (frame+1) and (frame+2), so that I know it is not noise
					// Check if the correspondence of first-second and second-third, is equivalent to the first-third one
					if (allMatches[i + 1][allMatches[i][j][0].trainIdx][0].trainIdx == matchesFirstThirdFrame[j][0].trainIdx) {

						// Good single 2d keypoint
						cv::Point2f point2d = keypointsAllImgs[i][allMatches[i][j][0].queryIdx].pt;
						cv::Point2f point2d_next_frame = keypointsAllImgs[i + 1][allMatches[i][j][0].trainIdx].pt;

						float depth1 = depthImages[i].at<uint16_t>(point2d);
						float depth2 = depthImages[i + 1].at<uint16_t>(point2d_next_frame);
						// Only points where depth is VALID
						if (depth1 > 400 && depth2 > 400 &&
							depth1 < 17000 && depth2 < 17000 && std::abs(depth1 - depth2) < 400) {
							good_matches[allMatches[i][j][0].queryIdx] = allMatches[i][j][0].trainIdx;
							good_matches_prev[allMatches[i][j][0].trainIdx] = allMatches[i][j][0].queryIdx;
						}

					}
				}
			}
		}
		// Set good keypoints of frame i
		pointClouds[i].setPoints2d(keypointsAllImgs[i]);

		// Set good matches between frame i and i+1
		pointClouds[i].setIndexMatchesFrames(good_matches);
		pointClouds[i + 1].setIndexPrevMatchesFrames(good_matches_prev);
	}
	
    // Visualize matches after removing outliers
    /*for(i=0;i<59;i++){
	    std::map<int, int> tmp_map_index_frame_zero_one = pointClouds[i].getIndexMatchesFrames();
	    std::vector<cv::DMatch> good_matches_img;

	    for (std::map<int,int>::iterator it = tmp_map_index_frame_zero_one.begin(); it != tmp_map_index_frame_zero_one.end(); ++it)
	    	good_matches_img.push_back(cv::DMatch(it->first, it->second, 0)); 

	    cv::Mat imgMatches;
	    cv::drawMatches(rgbImages[i], keypointsAllImgs[i], rgbImages[i+1], keypointsAllImgs[i+1], good_matches_img, imgMatches);

	    cv::imshow("Good Matches"+std::to_string(i), imgMatches);
	    cv::waitKey(0);
	    cv::destroyWindow("Good Matches" + std::to_string(i));
	}*/


	//****************DEBUG*************************: BACKPROJECTION
	/*for (int n = 0; n < 1; n++) {
		std::vector<Vector3f> global_3D_points;
		std::vector<cv::Vec3b> global_color_points;

		for (i = 0; i < 480; i++) {
			for (j = 0; j < 640; j++) {
				cv::Point2f point(j, i);
				Vector3f point_3D;

				if (depthImages[n].at<uint16_t>(point) > 0) {
					cv::Vec3b colors = rgbImages[n].at<cv::Vec3b>(i, j);

					float depth = (depthImages[n].at<uint16_t>(point) * 1.0f) / 5000.0f;
					point_3D = pointClouds[n].point2dToPoint3d(Vector2f(j, i), depth);
					global_3D_points.push_back(point_3D);
					global_color_points.push_back(colors);
				}
			}
		}

		generateOffFile("/offFiles/gt_new_712.off", global_3D_points, global_color_points, pointClouds); 
	}*/

    // 3D coordinates of keypoints without duplicates
    std::vector<Vector3f> global_3D_points;
    std::vector<cv::Vec3b> global_color_points;

	for (i = 0; i<FRAMES_TO_OPTIMIZE + WINDOW_SIZE; i++) {

		std::map<int, int> matchesFrames = pointClouds[i].getIndexMatchesFrames();
		std::map<int, int> keypointsTo3DGlobal = pointClouds[i].getGlobal3Dindices();

		for (auto& entry : matchesFrames) {
			int currentInd = entry.first;
			int nextInd = entry.second;

			// Search for points present in at least POINT_CONSECUTIVE_FRAMES frames
			int consFrames = consecutiveFrames(pointClouds, i, currentInd, 1);

			if (consFrames >= POINT_CONSECUTIVE_FRAMES) {

				// 3D POINT NOT IN GLOBAL ARRAY YET
				if (keypointsTo3DGlobal.find(currentInd) == keypointsTo3DGlobal.end()) {

					cv::Point2f keypoint_2D = keypointsAllImgs[i][currentInd].pt;
					Vector3f point_3D;

					// At the beginning, we use depth values from GT
					if (i == 0) {
						float depth = (depthImages[i].at<uint16_t>(keypoint_2D) * 1.0f) / 5000.0f;
						point_3D = pointClouds[i].point2dToPoint3d(Vector2f(keypoint_2D.x, keypoint_2D.y), depth);
					}
					else {
						// We will do triangulation later after optimization
						point_3D = Vector3f(MINF, MINF, MINF);
					}

					// Store inside point cloud the indices of the corresponding 3D points
					int indexThisFrame = currentInd;
					for (j = i; j < (consFrames + i); j++) {

						std::map<int, int> tmp_matches = pointClouds[j].getIndexMatchesFrames();

						pointClouds[j].appendGlobal3Dindices(indexThisFrame, global_3D_points.size());
						if (tmp_matches.find(indexThisFrame) == tmp_matches.end() && j != consFrames + i - 1) {
							std::vector<cv::DMatch> good_matches_img;

							std::map<int, int> prev = pointClouds[j].getIndexPrevMatchesFrames();
							good_matches_img.push_back(cv::DMatch(prev[indexThisFrame], indexThisFrame, 0));
							cv::Mat imgMatches;
							cv::drawMatches(rgbImages[j-1], keypointsAllImgs[j-1], rgbImages[j], keypointsAllImgs[j], good_matches_img, imgMatches);

							cv::imshow("Good Matches" + std::to_string(i), imgMatches);
							cv::waitKey(0);
							cv::destroyWindow("Good Matches" + std::to_string(i));
							std::cout << "i" << i << " " << j << " " << consFrames << " " << currentInd << " " << nextInd << std::endl;
							break;
						}
						indexThisFrame = tmp_matches[indexThisFrame];
					}

					// Append to global 3D vectors					
					global_color_points.push_back(rgbImages[i].at<cv::Vec3b>(keypoint_2D));
					global_3D_points.push_back(point_3D);
				}
			}
		}
	}
	// Optimise on poses and update on points
	Optimization optimizer;
	optimizer.setNbOfIterations(1);
	int breakingPoint = 0;
	for (i = 0; i < FRAMES_TO_OPTIMIZE + WINDOW_SIZE; i++) {
		// Add a break point
		if (i == FRAMES_TO_OPTIMIZE) {
			breakingPoint = global_3D_points.size();
		}
		Matrix4f finalPose = optimizer.estimatePose(pointClouds[i], global_3D_points);
		pointClouds[i + 1].setCameraExtrinsics(finalPose);
		std::map<int, int> match_2D_to_3D = pointClouds[i+1].getGlobal3Dindices();
		for (std::map<int, int>::iterator it = match_2D_to_3D.begin(); it != match_2D_to_3D.end(); ++it) {
			if (global_3D_points[it->second][0] == MINF) {
				cv::Point2f keypoint_2D = keypointsAllImgs[i + 1][it->first].pt;
				float depth = (depthImages[i + 1].at<uint16_t>(keypoint_2D) * 1.0f) / 5000.0f;
				Vector3f point_3D = pointClouds[i + 1].point2dToPoint3d(Vector2f(keypoint_2D.x, keypoint_2D.y), depth);
				global_3D_points[it->second] = point_3D;
			}
		}
	}

    std::cout << "PHASE 2: Optimization" << std::endl;
    for(i=0; i < FRAMES_TO_OPTIMIZE; i++){
    	optimizer.estimatePoseWithPoint(pointClouds, global_3D_points, i, WINDOW_SIZE);
        //triangulationWithLastFrame(pointClouds, global_3D_points, frames_to_consider - 1 + i);
    	std::cout << std::endl << std::endl << std::endl;
    }

	/*TEST TRIANGULATION
	std::map<int, int> tmp1 = pointClouds[0].getIndexMatchesFrames();

    std::vector<cv::KeyPoint> curr_2d = pointClouds[0].getPoints2d();
    std::vector<cv::KeyPoint> next_2d = pointClouds[1].getPoints2d();

    for (std::map<int,int>::iterator it = tmp1.begin(); it != tmp1.end(); ++it){

        cv::KeyPoint point2D_prev = curr_2d[it->first];
        std::vector<Vector2f> points2dFrame1;
        points2dFrame1.push_back(Vector2f(point2D_prev.pt.x, point2D_prev.pt.y));

        cv::KeyPoint point2D_curr = next_2d[it->second];
        std::vector<Vector2f> points2dFrame2;        
        points2dFrame2.push_back(Vector2f(point2D_curr.pt.x, point2D_curr.pt.y));        

        float dd = (depthImages[1].at<uint16_t>(point2D_curr.pt) * 1.0f) / 5000.0f;
        Vector3f test_result = pointClouds[1].point2dToPoint3d(Vector2f(point2D_curr.pt.x, point2D_curr.pt.y), dd);
        float dd = (depthImages[0].at<uint16_t>(point2D_prev.pt) * 1.0f) / 5000.0f;
        Vector3f test_result = pointClouds[0].point2dToPoint3d(Vector2f(point2D_prev.pt.x, point2D_prev.pt.y), dd);
        std::vector<Vector3f> tr_result = performTriangulation(points2dFrame1, points2dFrame2, pointClouds[0].getCameraExtrinsics(), pointClouds[1].getCameraExtrinsics(), pointClouds[0].getCameraIntrinsics(), test_result);

        std::cout << test_result << std::endl;
        std::cout << tr_result[0] << std::endl << std::endl;
    }*/

    generateOffFile("/offFiles/result_after_opt_final.off", global_3D_points, global_color_points, pointClouds, breakingPoint);

    return 0;
}


// Method to generate off file for 3d points
void generateOffFile(std::string filename, std::vector<Vector3f> points3d, std::vector<cv::Vec3b> colorValues, std::vector<PointCloud> pointClouds, int breakingPoint) {
    std::ofstream outFile(PROJECT_DIR + filename);
    if (!outFile.is_open()) return;
    if (points3d.size() == 0) return;
    
    // write header
    outFile << "COFF" << std::endl;
    outFile << points3d.size() << " 0 0" << std::endl;
    
    // Save camera position
    for (int i = 0; i < FRAMES_TO_OPTIMIZE; i++) {
        Matrix4f cameraExtrinsics = pointClouds[i].getCameraExtrinsics();
        Matrix3f rotation = cameraExtrinsics.block(0, 0, 3, 3);
        Vector3f translation = cameraExtrinsics.block(0, 3, 3, 1);
        Vector3f cameraPosition = -rotation.transpose()*translation;
        outFile << cameraPosition.x() << " " << cameraPosition.y() << " " << cameraPosition.z() << " 255 0 0" << std::endl;
    }

    // Save vertices
    for (int i = 0; i < breakingPoint; i++) {
        if (points3d[i].x() == MINF)
            outFile << "0 0 0 0 0 0" << std::endl;
    // OpenCV stores as BGR
        else
        	outFile << points3d[i].x() << " " << points3d[i].y() << " " << points3d[i].z() << " " << 
            static_cast<unsigned>(colorValues[i][2]) << " " << static_cast<unsigned>(colorValues[i][1]) << " " << static_cast<unsigned>(colorValues[i][0]) <<  std::endl;
    }
    outFile.close();
}


void get_data(std::string file_path, std::vector<cv::Mat>& depth_images, std::vector<cv::Mat>& rgb_images, std::vector<Matrix4f>& transformationMatrices){

    std::ifstream inFile;
    std::string fileName;
    
    inFile.open(DATA_PATH + file_path);

    if (!inFile) {
        std::cerr << "Unable to open file.\n" << std::endl;
        exit(1);
    }

    int i = 0;
    while (std::getline(inFile, fileName)) {
        i++;
        if (i % SKIP_FRAMES != 0)
        	continue;

        std::vector<std::string> current_line;
        split(fileName, ' ', current_line);

        cv::Mat depthImg = cv::imread(DATA_PATH + current_line[0], CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
        cv::Mat rgbImg = cv::imread(DATA_PATH + current_line[1], CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

        if (!depthImg.data || !rgbImg.data) {
            std::cout << "Image not found.\n" << std::endl;
            exit(1);
        }

        depth_images.push_back(depthImg);
        rgb_images.push_back(rgbImg);

        // Save poses
        Matrix4f transformationMatrix = getExtrinsicsFromQuaternion(current_line);
        transformationMatrices.push_back(transformationMatrix);     

        // TAKE FEW IMAGES DURING DEBUG
        if (i > 1000)
            break;
    }

    inFile.close();
}


int consecutiveFrames(std::vector<PointCloud>& PCs, int currPointCloudIndex, int currKeypointIndex, int currRecursion){
	
	std::map<int, int> matchesFrames = PCs[currPointCloudIndex].getIndexMatchesFrames();

	if (matchesFrames.find(currKeypointIndex) ==  matchesFrames.end())
		return currRecursion;
	
	else
		return consecutiveFrames(PCs, currPointCloudIndex + 1, matchesFrames[currKeypointIndex], currRecursion + 1);
}


Matrix4f getExtrinsicsFromQuaternion(std::vector<string> poses){
    int i, j;
    Matrix4f extrinsics;
    extrinsics.setIdentity();

    Eigen::Quaterniond q(Eigen::Vector4d(std::stod(poses[5]), std::stod(poses[6]), std::stod(poses[7]), std::stod(poses[8])));
    Eigen::Matrix<double, 3, 3> rotation = q.normalized().toRotationMatrix();

    for(i = 0; i < 3; i++){
        for(j = 0; j < 3; j++)
            extrinsics(i,j) = rotation(i,j);
    }
    extrinsics(0,3) = std::stod(poses[2]);
    extrinsics(1,3) = std::stod(poses[3]);
    extrinsics(2,3) = std::stod(poses[4]);

    return extrinsics;
}


std::vector<Vector3f> performTriangulation(std::vector<Vector2f> points2dFrame1, std::vector<Vector2f> points2dFrame2, Matrix4f extFrame1, Matrix4f extFrame2, Matrix3f camIntr){
    // Run the triangulation here
    // Find the matching points for which we do not have the 3d points
    // Triangulate them and add them to the point cloud

    std::cout << "PHASE 3.1: Running the triangulation" <<std::endl;

    // std::vector<Vector2f> points2dFrame1 stores the 2d coordinates of the points in the frame1
    // for which we do not have corresponding 3d point coordinate
    // but the 2d point is a good match w.r.t features

    // std::vector<Vector2f> points2dFrame2 stores the 2d coordinates of the points in the frame2
    // for which we do not have corresponding 3d point coordinate
    // but the 2d point is a good match w.r.t features

    // extrinsicsFrame1 stores the world to camera transformation for the frame1
    // extrinsicsFrame2 stores the world to camera transformation for the frame2
    // intrinsics stores the camera intrinsic parameters

    // This vector stores the 3D point coordinates that we will calculate during the triangulation
    std::vector<Vector3f> points3dTriangulation;

    if(points2dFrame2.size() > 0){

        // Defining Ax=b
        // Defining the A and b for the linear least square solver
        MatrixXf A(4, 3);
        A <<    0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0;

        VectorXf x(3);
        x <<    0.0, 0.0, 0.0;

        VectorXf b(4);
        b << 0.0, 0.0, 0.0, 0.0;

        std::cout << "PHASE 3.1: Number of points to triangulate: " << points2dFrame2.size() << std::endl;

        // Perform the triangulation to get the 3dPoints for points2dPreviousFrame and points2dCurrentFrame
        for( int g = 0; g < points2dFrame2.size(); g++){

            Vector2f pointframe1 = points2dFrame1[g];
            Vector2f pointframe2 = points2dFrame2[g];

            // Calculate the values to be inserted inside A and b for Ax=b
            A(0,0) = extFrame1(2,0) * pointframe1.x() - camIntr(0,0) * extFrame1(0,0) - camIntr(0,2) * extFrame1(2,0);
            A(0,1) = extFrame1(2,1) * pointframe1.x() - camIntr(0,0) * extFrame1(0,1) - camIntr(0,2) * extFrame1(2,1);
            A(0,2) = extFrame1(2,2) * pointframe1.x() - camIntr(0,0) * extFrame1(0,2) - camIntr(0,2) * extFrame1(2,2);

            A(1,0) = extFrame1(2,0) * pointframe1.y() - camIntr(1,1) * extFrame1(1,0) - camIntr(1,2) * extFrame1(2,0);
            A(1,1) = extFrame1(2,1) * pointframe1.y() - camIntr(1,1) * extFrame1(1,1) - camIntr(1,2) * extFrame1(2,1);
            A(1,2) = extFrame1(2,2) * pointframe1.y() - camIntr(1,1) * extFrame1(1,2) - camIntr(1,2) * extFrame1(2,2);


            A(2,0) = extFrame2(2,0) * pointframe2.x() - camIntr(0,0) * extFrame2(0,0) - camIntr(0,2) * extFrame2(2,0);
            A(2,1) = extFrame2(2,1) * pointframe2.x() - camIntr(0,0) * extFrame2(0,1) - camIntr(0,2) * extFrame2(2,1);
            A(2,2) = extFrame2(2,2) * pointframe2.x() - camIntr(0,0) * extFrame2(0,2) - camIntr(0,2) * extFrame2(2,2);

            A(3,0) = extFrame2(2,0) * pointframe2.y() - camIntr(1,1) * extFrame2(1,0) - camIntr(1,2) * extFrame2(2,0);
            A(3,1) = extFrame2(2,1) * pointframe2.y() - camIntr(1,1) * extFrame2(1,1) - camIntr(1,2) * extFrame2(2,1);
            A(3,2) = extFrame2(2,2) * pointframe2.y() - camIntr(1,1) * extFrame2(1,2) - camIntr(1,2) * extFrame2(2,2);


            b[0] = camIntr(0,0) * extFrame1(0,3) + camIntr(0,2) * extFrame1(2,3) - pointframe1.x() * extFrame1(2,3);
            b[1] = camIntr(1,1) * extFrame1(1,3) + camIntr(1,2) * extFrame1(2,3) - pointframe1.y() * extFrame1(2,3);

            b[2] = camIntr(0,0) * extFrame2(0,3) + camIntr(0,2) * extFrame2(2,3) - pointframe2.x() * extFrame2(2,3);
            b[3] = camIntr(1,1) * extFrame2(1,3) + camIntr(1,2) * extFrame2(2,3) - pointframe2.y() * extFrame2(2,3);

            // Solve the system of equations
            x = A.colPivHouseholderQr().solve(b);

            //std::cout << (A *  gt_p) << std::endl;
            //std::cout << b << std::endl;
            
            // Push back the coordinates of the newly calculated point
            points3dTriangulation.push_back(Vector3f(x.x(), x.y(), x.z()));

        }
        return points3dTriangulation;

    } else{
        std::cout << "PHASE 3.1: No New Points were present for Triangulation" <<std::endl;
        return points3dTriangulation;
    }
}


void triangulationWithLastFrame(std::vector<PointCloud>& pointClouds, std::vector<Vector3f>& global_3D_points, int ind_last_frame) {
    
	// Extract 2D points
    std::vector<cv::KeyPoint> secondLastFramePoints2d = pointClouds[ind_last_frame - 1].getPoints2d();
    std::vector<cv::KeyPoint> lastFramePoints2d = pointClouds[ind_last_frame].getPoints2d();

    // key => index 2D curr frame | value => index 2d prev frame
    std::map<int, int> prevIndexMatch = pointClouds[ind_last_frame].getIndexPrevMatchesFrames();

    // Mapping 2D and 3D global indices
    std::map<int,int> global3DIndices = pointClouds[ind_last_frame].getGlobal3Dindices();

    // Containers that store data for triangulation
    std::vector<int> minfIndices;
    std::vector<Vector2f> secondLastFramePoints;
    std::vector<Vector2f> lastFramePoints;

    for(auto& entry: global3DIndices) {
    	int pixelInd = entry.first;
    	int globalInd = entry.second;

    	// Accumulate points for triangulation
    	if(global_3D_points[globalInd][0] == MINF){
    		Vector2f point_A = Vector2f(secondLastFramePoints2d[prevIndexMatch[pixelInd]].pt.x, 
    			secondLastFramePoints2d[prevIndexMatch[pixelInd]].pt.y);
    		secondLastFramePoints.push_back(point_A);

    		Vector2f point_B = Vector2f(lastFramePoints2d[pixelInd].pt.x, lastFramePoints2d[pixelInd].pt.y);
    		lastFramePoints.push_back(point_B);

    		minfIndices.push_back(globalInd);
    	}
    }

    // Triangulation
    std::vector<Vector3f> restoredPoints = performTriangulation(secondLastFramePoints, lastFramePoints,
        pointClouds[ind_last_frame - 1].getCameraExtrinsics(), pointClouds[ind_last_frame].getCameraExtrinsics(),
        pointClouds[ind_last_frame].getCameraIntrinsics());

    // store the result into the global 3D points vector
    for(int t = 0; t < restoredPoints.size(); t++)   
        global_3D_points[minfIndices[t]] = restoredPoints[t];
}


void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}