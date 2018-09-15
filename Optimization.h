#include <ceres/ceres.h>
#include <ceres/rotation.h>


template <typename T>
class PoseConverter {
public:

	/**
	 *	Converts 4x4 transformation matrix to 3DOF
	 */
	static T* pose6DOF(const Matrix4f& rotationMatrix){
		T* pose = new T[3];

		T rotMatrix[9] = { rotationMatrix(0,0), rotationMatrix(1,0), rotationMatrix(2,0), rotationMatrix(0,1), 
			rotationMatrix(1,1), rotationMatrix(2,1), rotationMatrix(0,2), rotationMatrix(1,2), rotationMatrix(2,2)};
		ceres::RotationMatrixToAngleAxis(rotMatrix, pose);

		T* pose6DOF = new T[6];
		pose6DOF[0] = pose[0];
		pose6DOF[1] = pose[1];
		pose6DOF[2] = pose[2];
		pose6DOF[3] = rotationMatrix(0,3);
		pose6DOF[4] = rotationMatrix(1,3);
		pose6DOF[5] = rotationMatrix(2,3);

		return pose6DOF;
	}

	/**
	 * Converts the pose increment with rotation in SO3 notation and translation as 3D vector into
	 * transformation 4x4 matrix.
	 */
	static Matrix4f convertToMatrix(T* pose) {
		// pose[0,1,2] is angle-axis rotation.
		// pose[3,4,5] is translation.
		double* rotation = pose;
		double* translation = pose + 3;

		// Convert the rotation from SO3 to matrix notation (with column-major storage).
		double rotationMatrix[9];
		ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

		// Create the 4x4 transformation matrix.
		Matrix4f matrix;
		matrix.setIdentity();
		matrix(0, 0) = float(rotationMatrix[0]);	matrix(0, 1) = float(rotationMatrix[3]);	matrix(0, 2) = float(rotationMatrix[6]);	matrix(0, 3) = float(translation[0]);
		matrix(1, 0) = float(rotationMatrix[1]);	matrix(1, 1) = float(rotationMatrix[4]);	matrix(1, 2) = float(rotationMatrix[7]);	matrix(1, 3) = float(translation[1]);
		matrix(2, 0) = float(rotationMatrix[2]);	matrix(2, 1) = float(rotationMatrix[5]);	matrix(2, 2) = float(rotationMatrix[8]);	matrix(2, 3) = float(translation[2]);
		
		return matrix;
	}

	static T* pointVectorToPointer(const Vector3f& m_point) {
		T* point = new T[3];

		point[0] = m_point.x();
		point[1] = m_point.y();
		point[2] = m_point.z();

		return point;
	}
};

class Point3DTo2DConstraint {
public:
	Point3DTo2DConstraint(const Vector2f& targetPoint) :
		m_targetPoint{ targetPoint }
	{ }

	template <typename T>
	bool operator()(const T* const pose, const T* const point, T* residuals) const {
		
		const T* rotation = pose;
		const T* translation = pose + 3;

		// Rotation.
		T p[3];
		ceres::AngleAxisRotatePoint(rotation, point, p);

		// Translation.
		p[0] += translation[0]; 
		p[1] += translation[1]; 
		p[2] += translation[2];

		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Intrinsics
		T focal = T(525.0);
		T mx = T(319.5);
		T my = T(239.5);

		// Compute final projected point position.
		T predicted_x = focal * xp;
		T predicted_y = focal * yp;

		predicted_x += mx;
		predicted_y += my;

		// Important: Ceres automatically squares the cost function
		residuals[0] = predicted_x - T(m_targetPoint[0]);
		residuals[1] = predicted_y - T(m_targetPoint[1]);

		return true;
	}

	static ceres::CostFunction* create(const Vector2f& targetPoint) {
		return new ceres::AutoDiffCostFunction<Point3DTo2DConstraint, 2, 6, 3>(
			new Point3DTo2DConstraint(targetPoint)
		);
	}

protected:
	const Vector2f m_targetPoint;
};

class PointToPointPoseConstraint {
public:
	PointToPointPoseConstraint(const Vector3f& sourcePoint, const Vector2f& targetPoint) :
		m_sourcePoint{ sourcePoint },
		m_targetPoint{ targetPoint }
	{ }

	template <typename T>
	bool operator()(const T* const pose, T* residuals) const {
		// Important: Ceres automatically squares the cost function.

		const T* rotation = pose;
		const T* translation = pose + 3;

		// Rotation.
		T p[3];
		T point[] = { T(m_sourcePoint[0]), T(m_sourcePoint[1]), T(m_sourcePoint[2]) };
		ceres::AngleAxisRotatePoint(rotation, point, p);

		// Translation.
		p[0] += translation[0];
		p[1] += translation[1];
		p[2] += translation[2];

		// Intrinsics
		T fx = T(525.0);
		T fy = T(525.0);
		T mx = T(319.5);
		T my = T(239.5);

		T xp = fx * p[0];
		T yp = fy * p[1];

		// Compute final projected point position.
		T predicted_x = (xp / p[2]) + mx;
		T predicted_y = (yp / p[2]) + my;

		residuals[0] = predicted_x - T(m_targetPoint[0]);
		residuals[1] = predicted_y - T(m_targetPoint[1]);

		return true;
	}

	static ceres::CostFunction* create(const Vector3f& sourcePoint, const Vector2f& targetPoint) {
		return new ceres::AutoDiffCostFunction<PointToPointPoseConstraint, 2, 6>(
			new PointToPointPoseConstraint(sourcePoint, targetPoint)
			);
	}

protected:
	const Vector3f m_sourcePoint;
	const Vector2f m_targetPoint;
	const float LAMBDA = 0.1f;
};


class Point3DTo2DNoPoseConstraint {
public:
	Point3DTo2DNoPoseConstraint(const Vector2f& targetPoint, const Vector3f rotation, const Vector3f translation) :
		m_targetPoint{ targetPoint },
		m_rotation { rotation },
		m_translation { translation }
	{ }

	template <typename T>
	bool operator()(const T* const point, T* residuals) const {
		
		// Rotation.
		T rotation[] = {T(m_rotation[0]), T(m_rotation[1]), T(m_rotation[2])};
		T p[3];
		ceres::AngleAxisRotatePoint(rotation, point, p);

		// Translation.
		p[0] += T(m_translation[0]); 
		p[1] += T(m_translation[1]); 
		p[2] += T(m_translation[2]);

		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Intrinsics
		T focal = T(525.0);
		T mx = T(319.5);
		T my = T(239.5);

		// Compute final projected point position.
		T predicted_x = focal * xp;
		T predicted_y = focal * yp;

		predicted_x += mx;
		predicted_y += my;

		// Important: Ceres automatically squares the cost function
		residuals[0] = predicted_x - T(m_targetPoint[0]);
		residuals[1] = predicted_y - T(m_targetPoint[1]);

		return true;
	}

	static ceres::CostFunction* create(const Vector2f& targetPoint, const Vector3f rotation, const Vector3f translation) {
		return new ceres::AutoDiffCostFunction<Point3DTo2DNoPoseConstraint, 2, 3>(
			new Point3DTo2DNoPoseConstraint(targetPoint, rotation, translation)
		);
	}

protected:
	const Vector2f m_targetPoint;
	const Vector3f m_rotation;
	const Vector3f m_translation;
};



class Optimization {
public:
	Optimization() : 
		m_nIterations{ 10 }
	{ }

	void setNbOfIterations(unsigned nIterations) {
		m_nIterations = nIterations;
	}

	Matrix4f estimatePose(PointCloud pointCloud, std::vector<Vector3f>& global_3D_points) {
		Matrix4f estimatedPose = pointCloud.getCameraExtrinsics();
		double *pose = PoseConverter<double>::pose6DOF(estimatedPose);
		for (int i = 0; i < m_nIterations; ++i) {
			// Prepare constraints
			ceres::Problem problem;
			prepareConstraints(pointCloud, global_3D_points, pose, problem);

			// Configure options for the solver.
			ceres::Solver::Options options;
			configureSolver(options);

			// Run the solver (for one iteration).
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			std::cout << summary.BriefReport() << std::endl;
			//std::cout << summary.FullReport() << std::endl;
			Matrix4f matrix = PoseConverter<double>::convertToMatrix(pose);
			estimatedPose = matrix;
		}

		return estimatedPose;
	}

	void estimatePoseWithPoint(std::vector<PointCloud>& pointClouds, std::vector<Vector3f>& global_3D_points, const int starting_index, const int frames_to_consider) {

		double** poses = (double**)malloc(sizeof(double*) * frames_to_consider);
		for (int k = 0; k < frames_to_consider; k++)
			poses[k] = (double*)malloc(sizeof(double) * 6);

		double** points3D = (double**)malloc(sizeof(double*) * global_3D_points.size());
		for (int i = 0; i < global_3D_points.size(); i++) {
			points3D[i] = (double*)malloc(sizeof(double) * 3);
			points3D[i] = PoseConverter<double>::pointVectorToPointer(global_3D_points[i]);
		}
		std::cout << "Starting" << starting_index << std::endl;
		for (int i = 0; i < m_nIterations; ++i) {

			// Prepare constraints
			ceres::Problem problem;

			// Configure options for the solver.
			ceres::Solver::Options options;
			configureSolver(options);

			for(int j=starting_index; j < (starting_index + frames_to_consider); j++){

				Matrix4f estimatedPose = pointClouds[j].getCameraExtrinsics();
				if (estimatedPose.isIdentity()) {
					estimatedPose = pointClouds[j-1].getCameraExtrinsics();
				}
				poses[j - starting_index] = PoseConverter<double>::pose6DOF(estimatedPose);

				prepareConstraints(pointClouds[j], points3D, poses[j - starting_index], problem, starting_index, j);
			}

			// Run the solver (for one iteration)
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			std::cout << summary.BriefReport() << std::endl;
			//std::cout << summary.FullReport() << std::endl;

			for (int j = starting_index; j < (starting_index + frames_to_consider); j++) {
				Matrix4f updated_matrix = PoseConverter<double>::convertToMatrix(poses[j - starting_index]);
				pointClouds[j].setCameraExtrinsics(updated_matrix);
			}
		}

		//Update 3D points
		for (int i = 0; i < global_3D_points.size(); i++) {
			
			if(global_3D_points[i][0] != MINF)
				global_3D_points[i] = Vector3f(float(points3D[i][0]), float(points3D[i][1]), float(points3D[i][2]));
		}

		// Free memory
		for (int i = 0; i < frames_to_consider; i++)
			free(poses[i]);
		free(poses);

		for (int i = 0; i < global_3D_points.size(); i++)
			free(points3D[i]);
		free(points3D);
	}

private:
	unsigned m_nIterations;

	void configureSolver(ceres::Solver::Options& options) {
		// Ceres options.
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		options.use_nonmonotonic_steps = false;
		options.linear_solver_type = ceres::DENSE_QR;
		options.minimizer_progress_to_stdout = 1;
		options.max_num_iterations = 1000;
		options.num_threads = 8;
	}

	void prepareConstraints(PointCloud& pointCloud, double** global_3D_points, double* pose, ceres::Problem& problem, int startingFrame, int frameNumber) const {

		// We optimize on the transformation in SE3 notation: 3 parameters for the axis-angle vector of the rotation (its length presents
		// the rotation angle) and 3 parameters for the translation vector
		std::vector<cv::KeyPoint> points2D = pointCloud.getPoints2d();
		
		// KEY: 2D index feature, VALUE: index global 3D point vector
		std::map<int, int> match_2D_to_3D = pointCloud.getGlobal3Dindices();

		int validPointsPerFrame = 0;
		for (std::map<int,int>::iterator it = match_2D_to_3D.begin(); it != match_2D_to_3D.end(); ++it){

			// New point, will be added with triangulation after the optimization
			if (global_3D_points[it->second][0] == MINF || global_3D_points[it->second][2] == 0)
				continue;

			//if (global_3D_points[it->second][0] < -2.5 || global_3D_points[it->second][0] > 1.2 || global_3D_points[it->second][1] < -4 ||
			//	global_3D_points[it->second][1] > 1 || global_3D_points[it->second][2] < 0) {
			//	std::cout << it->second << std::endl;
			//	std::cout << frameNumber << std::endl;
			//	std::cout << global_3D_points[it->second][0] << " " << global_3D_points[it->second][1] << " " << global_3D_points[it->second][2] << std::endl;
			//	global_3D_points[it->second][0] = MINF;
			//	global_3D_points[it->second][1] = MINF;
			//	global_3D_points[it->second][2] = MINF;
			//	continue;
			//}
			
			cv::Point2f tmp = points2D[it->first].pt;
			const auto& targetPoint = Vector2f(tmp.x, tmp.y);

			if (!targetPoint.allFinite()){
				std::cout << "SHOULD NOT ENTER HERE" << std::endl;
				continue;
			}

			validPointsPerFrame++;
			if (frameNumber == startingFrame){
				Vector3f rotation = Vector3f(pose[0], pose[1], pose[2]);
				Vector3f translation = Vector3f(pose[3], pose[4], pose[5]);
				
				problem.AddResidualBlock(Point3DTo2DNoPoseConstraint::create(targetPoint, rotation, translation), NULL, global_3D_points[it->second]);
			}
			else
				problem.AddResidualBlock(Point3DTo2DConstraint::create(targetPoint), NULL, pose, global_3D_points[it->second]);
		}

		if (validPointsPerFrame < 15) {
			std::cout << "Too few points" << std::endl;
			std::cout << validPointsPerFrame << std::endl;
		}
	}

	// For estimation of pose only
	void prepareConstraints(PointCloud& pointCloud, std::vector<Vector3f>& global_3D_points, double* pose, ceres::Problem& problem) const {

		std::vector<cv::KeyPoint> points2D = pointCloud.getPoints2d();
		// KEY: 2D index feature, VALUE: index global 3D point vector
		std::map<int, int> match_2D_to_3D = pointCloud.getGlobal3Dindices();

		for (std::map<int, int>::iterator it = match_2D_to_3D.begin(); it != match_2D_to_3D.end(); ++it) {

			if (global_3D_points[it->second][0] == MINF || global_3D_points[it->second][2] == 0)
				continue;
			const auto& sourcePoint = global_3D_points[it->second];
			cv::Point2f tmp = points2D[it->first].pt;
			const auto& targetPoint = Vector2f(tmp.x, tmp.y);

			if (!targetPoint.allFinite()) {
				std::cout << "SHOULD NOT ENTER HERE" << std::endl;
				continue;
			}
			problem.AddResidualBlock(PointToPointPoseConstraint::create(sourcePoint, targetPoint), NULL, pose);
		}

		return;
	}

};
