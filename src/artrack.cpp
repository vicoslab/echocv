
#include <echolib/opencv.h>

#include <memory>
#include <experimental/filesystem>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;
using namespace echolib;

#define PATTERN_SIZE 100

Matx44f rotationMatrix(float x, float y, float z)
{
	Matx44f RX = Matx44f(
	                 1, 0,      0,       0,
	                 0, cos(x), -sin(x), 0,
	                 0, sin(x),  cos(x), 0,
	                 0, 0,       0,     1);

	Matx44f RY = Matx44f(
	                 cos(y), 0, -sin(y), 0,
	                 0, 1,          0, 0,
	                 sin(y), 0,  cos(y), 0,
	                 0, 0,          0, 1);

	Matx44f RZ = Matx44f(
	                 cos(z), -sin(z), 0, 0,
	                 sin(z),  cos(z), 0, 0,
	                 0,          0,           1, 0,
	                 0,          0,           0, 1);

	return RX * RY * RZ;

}

Matx44f translationMatrix(float x, float y, float z)
{
	return Matx44f(
	           1, 0, 0, x,
	           0, 1, 0, y,
	           0, 0, 1, z,
	           0, 0, 0, 1);

}

Point3f extractHomogeneous(Matx41f hv)
{
	Point3f f = Point3f(hv(0, 0) / hv(3, 0), hv(1, 0) / hv(3, 0), hv(2, 0) / hv(3, 0));
	return f;
}

class Pattern {
public:
	Pattern(int id, float size, const string& filename, const string& name, Matx44f offset = Matx44f::eye()) :
		size(size), name(name), offset(offset) {

		char* buffer;
		size_t buffer_size;

		Mat img = imread(filename, 0);

		if (img.cols != img.rows) {
			throw runtime_error("Not a square pattern");
		}

		int msize = PATTERN_SIZE;

		Mat src(msize, msize, CV_8UC1);
		Point2f center((msize - 1) / 2.0f, (msize - 1) / 2.0f);
		Mat rot_mat(2, 3, CV_32F);

		resize(img, src, Size(msize, msize));
		Mat subImg = src(Range(msize / 4, 3 * msize / 4), Range(msize / 4, 3 * msize / 4));
		markers.push_back(subImg);

		rot_mat = getRotationMatrix2D(center, 90, 1.0);

		for (int i = 1; i < 4; i++) {
			Mat dst = Mat(msize, msize, CV_8UC1);
			rot_mat = getRotationMatrix2D(center, -i * 90, 1.0);
			warpAffine(src, dst, rot_mat, Size(msize, msize));
			Mat subImg = dst(Range(msize / 4, 3 * msize / 4), Range(msize / 4, 3 * msize / 4));
			markers.push_back(subImg);
		}

	}

	~Pattern() {};

	Matx44f getOffset() {
		return offset;
	}

	float getSize()  {
		return size;
	}

	string getName() {
		return name;
	}

	double match(const Mat& src, int& orientation) {

		int i;
		double tempsim;
		double N = (double)(PATTERN_SIZE * PATTERN_SIZE / 4);
		double nom, den;

		Scalar mean_ext, std_ext, mean_int, std_int;

		Mat interior = src(cv::Range(PATTERN_SIZE / 4, 3 * PATTERN_SIZE / 4), cv::Range(PATTERN_SIZE / 4, 3 * PATTERN_SIZE / 4));

		meanStdDev(src, mean_ext, std_ext);
		meanStdDev(interior, mean_int, std_int);

		//printf("ext: %f int: %f \n", mean_ext.val[0], mean_int.val[0]);

		if ((mean_ext.val[0] > mean_int.val[0]))
			return -1;

		double normSrcSq = pow(norm(interior), 2);

		//zero_mean_mode;
		int zero_mean_mode = 1;

		//use correlation coefficient as a robust similarity measure
		double confidence = -1.0;
		for (i = 0; i < markers.size(); i++) {

			double const nnn = pow(norm(markers.at(i)), 2);

			if (zero_mean_mode == 1) {

				double const mmm = mean(markers.at(i)).val[0];

				nom = interior.dot(markers.at(i)) - (N * mean_int.val[0] * mmm);
				den = sqrt( (normSrcSq - (N * mean_int.val[0] * mean_int.val[0]) ) * (nnn - (N * mmm * mmm)));
				tempsim = nom / den;
			}
			else
			{
				tempsim = interior.dot(markers.at(i)) / (sqrt(normSrcSq * nnn));
			}

			if (tempsim > confidence) {
				confidence = tempsim;
				orientation = i;
			}
		}

		return confidence;
	}


private:

	int id;
	float size;
	string name;
	std::vector<cv::Mat> markers;
	cv::Mat offset;

};


class PatternDetection
{
public:
	PatternDetection(shared_ptr<Pattern> pattern, const Mat& rotVec, const Mat& transVec,
	                 double confidence, vector<Point2f> corners) : pattern(pattern),
		confidence(confidence), rotVec(rotVec), transVec(transVec) {

		orientation = -1;
		this->corners[0] = corners.at(0);
		this->corners[1] = corners.at(1);
		this->corners[2] = corners.at(2);
		this->corners[3] = corners.at(3);
	}

	~PatternDetection() {};

	void draw(Mat& frame, const Mat& camMatrix, const Mat& distMatrix) {

		CvScalar color = cvScalar(255, 0, 255);

		float size = pattern->getSize();

		Point3f origin = extractHomogeneous(pattern->getOffset().inv() * Scalar(0, 0, 0, 1));
		std::vector<cv::Point2f> model2ImagePts;
		//model 3D points: they must be projected to the image plane
		Mat modelPts = (Mat_<float>(5, 3) <<
		                0, 0, 0,
		                size, 0, 0,
		                0, size, 0,
		                0, 0, size,
		                origin.x, origin.y, origin.z );

		/* project model 3D points to the image. Points through the transformation matrix
		(defined by rotVec and transVec) are "transfered" from the pattern CS to the
		camera CS, and then, points are projected using camera parameters
		(camera matrix, distortion matrix) from the camera 3D CS to its image plane
		*/
		projectPoints(modelPts, rotVec, transVec, camMatrix, distMatrix, model2ImagePts);

		cv::line(frame, model2ImagePts.at(0), model2ImagePts.at(4), cvScalar(255, 0, 255), 1);
		cv::line(frame, model2ImagePts.at(0), model2ImagePts.at(1), cvScalar(0, 0, 255), 3);
		cv::line(frame, model2ImagePts.at(0), model2ImagePts.at(2), cvScalar(0, 255, 0), 3);
		cv::line(frame, model2ImagePts.at(0), model2ImagePts.at(3), cvScalar(255, 0, 0), 3);

		cv::putText(frame, pattern->getName(), model2ImagePts.at(0), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 155, 0), 1);

		model2ImagePts.clear();

	}

	shared_ptr<Pattern> getPattern() {
		return pattern;
	}

	void getExtrinsics(Mat& rotation, Mat& translation) {
		Rodrigues(rotVec, rotation);

		transVec.copyTo(translation);
	}

	double getConfidence() {
		return confidence;
	}

	Point2f getCorner(int i)
	{
		return corners[i];
	}

private:

	shared_ptr<Pattern> pattern;
	int orientation; //{0,1,2,3}
	double confidence; //min: -1, max: 1
	Mat rotVec, transVec;
	Point2f corners[4];

};

class PatternDetector
{
public:
	PatternDetector(double threshold = 5, int block_size = 45, double conf_threshold = 0.60)  {


		this->threshold = threshold;//for image thresholding
		this->block_size = block_size;//for adaptive image thresholding
		this->confThreshold = conf_threshold;//bound for accepted similarities between detected patterns and loaded patterns
		normROI = Mat(PATTERN_SIZE, PATTERN_SIZE, CV_8UC1);//normalized ROI

		//corner of normalized area
		norm2DPts[0] = Point2f(0, 0);
		norm2DPts[1] = Point2f(PATTERN_SIZE - 1, 0);
		norm2DPts[2] = Point2f(PATTERN_SIZE - 1, PATTERN_SIZE - 1);
		norm2DPts[3] = Point2f(0, PATTERN_SIZE - 1);

	}

	~PatternDetector() {};

	int loadPattern(const string& filename, const string& name, double realsize = 50, Matx44f offset = Matx44f())  {

		library.push_back(make_shared<Pattern>(library.size(), realsize, filename, name, offset));

		offsets.push_back(offset);

		return (int) (library.size() - 1);
	}

	//detect patterns in the input frame
	void detect(const Mat &frame, const Mat& cameraMatrix, const Mat& distortions, vector<PatternDetection>& foundPatterns) {

		Point2f roi2DPts[4];
		Mat binImage2;

		//binarize image
		convertAndBinarize(frame, binImage, grayImage);
		binImage.copyTo(binImage2);

		int avsize = (binImage.rows + binImage.cols) / 2;

		vector<vector<Point> > contours;
		vector<Point> polycont;

		//find contours in binary image
		cv::findContours(binImage2, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

		unsigned int i;
		Point p;
		int pMinX, pMinY, pMaxY, pMaxX;

		for (i = 0; i < contours.size(); i++) {
			Mat contourMat = Mat (contours[i]);
			const double per = arcLength( contourMat, true);
			//check the perimeter
			if (per > (avsize / 4) && per < (4 * avsize)) {
				polycont.clear();
				approxPolyDP( contourMat, polycont, per * 0.02, true);

				//check rectangularity and convexity
				if (polycont.size() == 4 && isContourConvex(Mat (polycont))) {

					//locate the 2D box of contour,
					p = polycont.at(0);
					pMinX = pMaxX = p.x;
					pMinY = pMaxY = p.y;
					int j;
					for (j = 1; j < 4; j++) {
						p = polycont.at(j);
						if (p.x < pMinX) {
							pMinX = p.x;
						}
						if (p.x > pMaxX) {
							pMaxX = p.x;
						}
						if (p.y < pMinY) {
							pMinY = p.y;
						}
						if (p.y > pMaxY) {
							pMaxY = p.y;
						}
					}
					Rect box(pMinX, pMinY, pMaxX - pMinX + 1, pMaxY - pMinY + 1);

					//find the upper left vertex
					double d;
					double dmin = (4 * avsize * avsize);
					int v1 = -1;
					for (j = 0; j < 4; j++) {
						d = norm(polycont.at(j));
						if (d < dmin) {
							dmin = d;
							v1 = j;
						}
					}

					//store vertices in refinedVertices and enable sub-pixel refinement if you want
					vector<Point2f> refinedVertices;
					refinedVertices.clear();
					for (j = 0; j < 4; j++) {
						refinedVertices.push_back(polycont.at(j));
					}

					//refine corners
					cornerSubPix(grayImage, refinedVertices, Size(3, 3), Size(-1, -1), TermCriteria(1, 3, 1));

					//rotate vertices based on upper left vertex; this gives you the most trivial homogrpahy
					for (j = 0; j < 4; j++) {
						roi2DPts[j] = Point2f(refinedVertices.at((4 + v1 - j) % 4).x - pMinX, refinedVertices.at((4 + v1 - j) % 4).y - pMinY);
					}

					//normalize the ROI (find homography and warp the ROI)
					normalizePattern(grayImage, roi2DPts, box, normROI);

					double confidence = 0;
					int orientation;

					int id = identifyPattern(normROI, confidence, orientation);

					//push-back pattern in the stack of foundPatterns and find its extrinsics
					if (id >= 0) {

						vector<Point2f> candidateCorners;
						Mat rotVec = (Mat_<float>(3, 1) << 0, 0, 0);
						Mat transVec = (Mat_<float>(3, 1) << 0, 0, 0);

						for (j = 0; j < 4; j++) {
							candidateCorners.push_back(refinedVertices.at((8 - orientation + v1 - j) % 4));
						}

						//find the transformation (from camera CS to pattern CS)
						calculateExtrinsics(library[id]->getSize(), cameraMatrix, distortions, rotVec, transVec, candidateCorners);

						PatternDetection detected_pattern(library[id], rotVec, transVec, confidence, candidateCorners);
						foundPatterns.push_back(detected_pattern);

					}
				}
			}
		}
	}

private:

	//solves the exterior orientation problem between patten and camera
	void calculateExtrinsics(const float size, const Mat& cameraMatrix, const Mat& distortions, Mat& rotVec, Mat& transVec, const vector<Point2f>& imagePoints) {
		//3D points in pattern coordinate system
		vector<Point3f> objectPoints;
		objectPoints.push_back(Point3f(-size / 2, -size / 2, 0));
		objectPoints.push_back(Point3f(size / 2, -size / 2, 0));
		objectPoints.push_back(Point3f(size / 2, size / 2, 0));
		objectPoints.push_back(Point3f(-size / 2, size / 2, 0));

		solvePnP(objectPoints, imagePoints, cameraMatrix, distortions, rotVec, transVec);

		rotVec.convertTo(rotVec, CV_32F);
		transVec.convertTo(transVec, CV_32F);
	}

	void convertAndBinarize(const Mat& src, Mat& dst1, Mat& dst2)
	{

		//dst1: binary image
		//dst2: grayscale image

		if (src.channels() == 3) {
			cvtColor(src, dst2, CV_BGR2GRAY);
		}
		else {
			src.copyTo(dst2);
		}

		adaptiveThreshold( dst2, dst1, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, block_size, threshold);

		dilate(dst1, dst1, Mat());
	}

	void normalizePattern(const Mat& src, const Point2f roiPoints[], Rect& rec, Mat& dst) {

		//compute the homography
		Mat homography = getPerspectiveTransform( roiPoints, norm2DPts);
		cv::Mat subImg = src(cv::Range(rec.y, rec.y + rec.height), cv::Range(rec.x, rec.x + rec.width));

		//warp the input based on the homography model to get the normalized ROI
		cv::warpPerspective( subImg, dst, homography, Size(dst.cols, dst.rows));

	}
	int identifyPattern(const Mat& src, double& confidence, int& orientation) {
		if (library.size() < 1) {
			return -1;
		}

		unsigned int j;
		orientation = 0;
		int match = -1;

		//use correlation coefficient as a robust similarity measure
		confidence = confThreshold;
		for (j = 0; j < library.size(); j++) {

			double m = library[j]->match(src, orientation);

			if (m > confidence) {
				confidence = m;
				match = j;
			}

		}

		return match;

	}

	std::vector<shared_ptr<Pattern> > library;
	std::vector<Matx44f> offsets;

	int block_size;
	double confThreshold, threshold;

	Mat binImage, grayImage, normROI;
	Point2f norm2DPts[4];

};


bool debug = false;
Mat debug_image;

CameraExtrinsics location;
CameraIntrinsics parameters;
SharedTypedPublisher<CameraExtrinsics> location_publisher;
SharedTypedSubscriber<CameraIntrinsics> parameters_listener;
PatternDetector detector;

bool localize_camera(vector<PatternDetection> detections) {

	if (detections.size() < 1) return false;

	PatternDetection* anchor = NULL;

	Mat rotVec, transVec;

	vector<Point3f> surfacePoints;
	vector<Point2f> imagePoints;

	Matx44f global = Matx44f::eye(); // rotationMatrix(0, 0, M_PI);

	for (unsigned int i = 0; i < detections.size(); i++) {

		float size = (float) detections[i].getPattern()->getSize();
		Matx44f transform = detections[i].getPattern()->getOffset();

		surfacePoints.push_back(extractHomogeneous(global * transform * Scalar(-size / 2, -size / 2, 0, 1)));
		surfacePoints.push_back(extractHomogeneous(global * transform * Scalar(size / 2, -size / 2, 0, 1)));
		surfacePoints.push_back(extractHomogeneous(global * transform * Scalar(size / 2, size / 2, 0, 1)));
		surfacePoints.push_back(extractHomogeneous(global * transform * Scalar(-size / 2, size / 2, 0, 1)));

		imagePoints.push_back(detections.at(i).getCorner(0));
		imagePoints.push_back(detections.at(i).getCorner(1));
		imagePoints.push_back(detections.at(i).getCorner(2));
		imagePoints.push_back(detections.at(i).getCorner(3));
	}

	//if (surfacePoints.size() > 4) {
	//	DEBUGMSG("Estimating plane on %d points\n", (int) surfacePoints.size());
	//	solvePnPRansac(surfacePoints, imagePoints, intrinsics, distortion, rotVec, translation, false, 100, 13.0, std::max(8, (int) surfacePoints.size() / 2));
	//} else {
	solvePnP(surfacePoints, imagePoints, Mat(parameters.intrinsics), parameters.distortion, rotVec, transVec);
	//}
	rotVec.convertTo(rotVec, CV_32F);
	transVec.convertTo(location.translation, CV_32F);
	Rodrigues(rotVec, location.rotation);

	/*
		Mat extrinsics;
		hconcat(rotation, translation, extrinsics);
		Mat projection = intrinsics * extrinsics;
		projection.col(0).copyTo(homography.col(0));
		projection.col(1).copyTo(homography.col(1));
		projection.col(3).copyTo(homography.col(2));
	*/

	return true;
}

void handle_frame(Mat& image) {

	if (image.empty())
		return;

	vector<PatternDetection> detectedPatterns;
	detector.detect(image, Mat(parameters.intrinsics), parameters.distortion, detectedPatterns);

	if (debug) {
		image.copyTo(debug_image);
		for (size_t i = 0; i < detectedPatterns.size(); i++) {
			detectedPatterns.at(i).draw(debug_image, Mat(parameters.intrinsics), parameters.distortion);
		}
		imshow("AR Track", debug_image);
	}

	if (detectedPatterns.size() > 0) {

		localize_camera(detectedPatterns);

		location_publisher->send(location);
		// TODO: publish

	}
}

using namespace std::experimental::filesystem::v1;

int main(int argc, char** argv) {

	if (argc < 2) {
		cerr << "No AR board description specified." << endl;
		exit(-1);
	}

	FileStorage fs(argv[1], FileStorage::READ);

	if (!fs.isOpened()) {
		cerr << "Invalid AR board description." << endl;
		exit(-1);
	}

	path descriptor_path = path(argv[1]);

	for (FileNodeIterator it = fs.root().begin(); it != fs.root().end(); ++it) {
		int template_size;
		float template_orientation, rx, ry, rz, ox, oy, oz;

		read((*it)["size"], template_size, 50);
		read((*it)["rotation"]["x"], rx, 0);
		read((*it)["rotation"]["y"], ry, 0);
		read((*it)["rotation"]["z"], rz, 0);
		read((*it)["origin"]["x"], ox, 0);
		read((*it)["origin"]["y"], oy, 0);
		read((*it)["origin"]["z"], oz, 0);

		rx = M_PI * (rx) / 180;
		ry = M_PI * (ry) / 180;
		rz = M_PI * (rz) / 180;

		Matx44f offset = rotationMatrix(rx, ry, rz) * translationMatrix(ox, oy, oz);

		path filename = string((*it).name());
		string template_file = filename.is_relative() ? descriptor_path.parent_path() / filename : filename;

		string template_name = string((*it)["name"]);
		if (template_name.empty())
			template_name = string(filename.filename());

		detector.loadPattern(template_file, template_name, template_size, offset);

	}

	IOLoop loop;

	SharedClient client = echolib::connect(loop);

	location.rotation = Matx33f::eye();
	location.translation = (Matx31f::zeros());

	parameters_listener = make_shared<TypedSubscriber<CameraIntrinsics> >(client, "intrinsics",
	[](shared_ptr<CameraIntrinsics> param) {
		parameters = *param;
		parameters_listener.reset();
	});

	ImageSubscriber sub(client, "camera", handle_frame);
	location_publisher = make_shared<TypedPublisher<CameraExtrinsics> >(client, "location");

	while (loop.wait(100)) {
		if (debug) waitKey(1);
	}

	exit(0);
}


