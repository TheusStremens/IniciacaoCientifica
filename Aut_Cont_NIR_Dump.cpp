#include <iostream>
#include <fstream>
#include <sstream>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include <zlib.h>
//#include <lzf.h>
#include "liblzf/lzf.h"
#include "liblzf/lzf_d.c"
#include <linux/input.h>
#include "opencv/cv.h"
#include "opencv2/videoio.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/video/tracking.hpp>

const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.45;         // Should be atleast 0.5
const double FACE_ELLIPSE_H = 0.80;     //0.80
const double DESIRED_LEFT_EYE_X = 0.26;     // Controls how much of the face is visible after preprocessing. 0.16 and 0.14
const double DESIRED_LEFT_EYE_Y = 0.24;
const int MAX_FRAME_LOST = 4;
const int BORDER = 8;  // Border between GUI elements to the edge of the image.
const int FACE_SIZE = 200;
//parametros reconhecimento continuo
const float Usafe = 15.8142; //30.0509;
const float UnotSafe = 47.7215; //95.6884;
const float Rsafe = 8.02247; //24.8666;
const float RnotSafe = 6.05694; //30.7036;

const float E = 2.71828182845904523536;
const float LN2 = 0.693147180559945309417;
const float K_DROP = 15.0;
const int FRAMES_LOGIN = 5;
const int WINDOWN = 40;
const int MAXMODELO = 30;
const float THR_QUEDA = 0.10;

using namespace cv;
using namespace cv::face;
using namespace std;

const string PATH_CASCADE_FACE = "/home/matheusm/Cascades/cascade.xml";
const string PATH_CASCADE_RIGHTEYE = "/home/matheusm/Cascades/haarcascade_mcs_righteye_alt.xml";
const string PATH_CASCADE_LEFTEYE = "/home/matheusm/Cascades/haarcascade_mcs_lefteye_alt.xml";
const string PATH_CASCADE_BOTHEYES = "/home/matheusm/Cascades/haarcascade_eye.xml";
//const string PATH_CSV_FACES = "/home/matheusm/framesVideoSujeito4.txt";

int framesLost = 0;
KalmanFilter KFrightEye, KFleftEye;
Mat_<float> state; /* (x, y, Vx, Vy) */
Mat processNoise;
Mat_<float> measurement;
bool initiKalmanFilter = true;
Point leftEyePredict = Point(-1,-1);
Point rightEyePredict = Point(-1,-1);

extern "C" void __cxa_pure_virtual(void)
{
   std::cout << "__cxa_pure_virtual: pure virtual method called" << std::endl << std::flush;
}

enum CompressionFormat { NONE = 0x00000000, ZLIB = 0x00000001, LIBLZF = 0x00000002 };
enum Mode { VIDEO_RGB_24B = 0x00000000, VIDEO_YUYV_16B = 0x00000001, VIDEO_IR_24B = 0x00000002, VIDEO_DEPTH_16B = 0x00000003, VIDEO_DIR_24B = 0x00000004, DATA_MOUSE = 0x00000005, DATA_KEYBOARD = 0x00000006};

typedef struct __stream {
  FILE *f;
  u_int32_t mode, compression;
  int width, height, size;
  Mat color, depth, infrared, tmp;
  unsigned char *buf1, *buf2;
  struct timeval t;
} STREAM;

typedef struct __data {
  FILE *f;
  u_int32_t mode;
  struct input_event ev;
  int coord[2];
} DATA;

/* Create data stream */
DATA * createData(char *filename) {
  DATA *r;

  r = new DATA;
  r->f = fopen(filename, "r");

  fread(&r->mode, sizeof(u_int32_t), 1, r->f);

  return r;
}

/* Release data stream */
void releaseData(DATA *d) {
  fclose(d->f);
  delete d;
}

/* Next data event */
bool dataNext(DATA *d) {
  if(fread((void *) &d->ev, sizeof(struct input_event), 1, d->f) != 1)
    return false;
  if(d->ev.type == 2 && (d->ev.code == 0 || d->ev.code == 1))
    fread((void *) d->coord, sizeof(int), 2, d->f);
}

/* Create video stream */
STREAM *createStream(char *filename) {
  STREAM *r;

  r = new STREAM;
  r->f = fopen(filename, "r");

  fread(&r->mode, sizeof(u_int32_t), 1, r->f);
  switch(r->mode) {
    case VIDEO_RGB_24B:
    case VIDEO_YUYV_16B:
    case VIDEO_DEPTH_16B:
    case VIDEO_IR_24B:
    case VIDEO_DIR_24B:
      fread(&r->compression, sizeof(u_int32_t), 1, r->f);
      fread(&r->width, sizeof(int), 1, r->f);
      fread(&r->height, sizeof(int), 1, r->f);
      switch(r->mode) {
        case VIDEO_YUYV_16B:
          r->size = r->width*r->height*2;
          r->tmp.create(r->height, r->width, CV_8UC2);
          r->color.create(r->height, r->width, CV_8UC3);
          break;
        case VIDEO_DEPTH_16B:
          r->size = r->width*r->height*2;
          r->depth.create(r->height, r->width, CV_16U);
          break;
        case VIDEO_RGB_24B:
          r->size = r->width*r->height*4;
          r->color.create(r->height, r->width, CV_8UC4);
          break;
        case VIDEO_DIR_24B:
          r->size = r->width*r->height*8;
          r->depth.create(r->height, r->width, CV_32FC1);
//          r->infrared.create(r->height, r->width, CV_8U);
          break;
        case VIDEO_IR_24B:
          r->size = r->width*r->height*8;
          r->infrared.create(r->height, r->width, CV_32FC1);
          break;
      }
      r->buf1 = new unsigned char[r->size];
      r->buf2 = new unsigned char[r->size];
      break;
  }

  return r;
}

/* Release video stream */
void releaseStream(STREAM *s) {
  switch(s->mode) {
    case VIDEO_RGB_24B:
      s->color.release();
      break;
    case VIDEO_YUYV_16B:
      s->tmp.release();
      s->color.release();
      break;
    case VIDEO_DEPTH_16B:
      s->depth.release();
      break;
    case VIDEO_DIR_24B:
      s->depth.release();
//      s->infrared.release();
      break;
    case VIDEO_IR_24B:
      s->infrared.release();
      break;
  }
  fclose(s->f);
  delete s;
}

/* Next video frame */
bool streamNext(STREAM *s) {

  /* Read timestamp */
  if(fread(&s->t, sizeof(struct timeval), 1, s->f) != 1)
    return false; 
  /* Read and decompress next frame */
  switch(s->compression) {
    case NONE:
      if(fread(s->buf1, sizeof(char), s->size, s->f) != s->size)
        return false;
      break;
    case ZLIB:
      /* NOT IMPLEMENTED YET */
      break;
    case LIBLZF:
      unsigned long int csize, nsize;
      if(fread(&csize, sizeof(unsigned long int), 1, s->f) != 1)
        return false;
//std::cout<<fread(s->buf2, sizeof(char), csize, s->f)<<" <- NUMBER -> "<<csize<<endl;
      if(fread(s->buf2, sizeof(char), csize, s->f) != csize){
        return false;}

      switch(s->compression) {
        case ZLIB:
          break;
        case LIBLZF:
          nsize = lzf_decompress(s->buf2, csize, s->buf1, s->size);
          break;
      }
  }


  /* Convert frame to OpenCV */
  switch(s->mode) {
    case VIDEO_RGB_24B:
      s->color = cv::Mat(s->height, s->width, CV_8UC4, s->buf1);
      break;
    case VIDEO_YUYV_16B:
      for(int j = 0 ; j < s->height ; j++){
        int step16 = s->width*2*j;
        for(int i = 0 ; i < s->width ; i++){
          int pixel16 = step16 + 2*i;
          s->tmp.at<u_int16_t>(j,i) = *(u_int16_t *)(s->buf1 + pixel16);
        }
      }
      cvtColor(s->tmp,s->color,COLOR_YUV2BGR_YUYV);

      break;
    case VIDEO_DEPTH_16B:
      /* NOT IMPLEMENTED YET */
      break;
    case VIDEO_DIR_24B:
      s->depth = cv::Mat(s->height, s->width, CV_32FC1, s->buf1)/ 4500.0f;
      break;
    case VIDEO_IR_24B:
      s->infrared = cv::Mat(s->height, s->width, CV_32FC1, s->buf1) / 80000.0f;
      break;
  }

  return true;
}

Mat faceNormalize(Mat face, int desiredFaceWidth, bool &sucess) {
  int im_height = face.rows;
  Mat faceProcessed;

  cv::resize(face, face, Size(400, 400), 1.0, 1.0, INTER_CUBIC);

  const float EYE_SX = 0.16f;
  const float EYE_SY = 0.26f;
  const float EYE_SW = 0.30f;
  const float EYE_SH = 0.28f;

  int leftX = cvRound(face.cols * EYE_SX);
  int topY = cvRound(face.rows * EYE_SY);
  int widthX = cvRound(face.cols * EYE_SW);
  int heightY = cvRound(face.rows * EYE_SH);
  int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW) );  

  Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
  Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));

  CascadeClassifier haar_cascade;
  haar_cascade.load(PATH_CASCADE_LEFTEYE);

  vector< Rect_<int> > detectedRightEye;
  vector< Rect_<int> > detectedLeftEye;
  Point leftEye = Point(-1, -1), rightEye = Point(-1, -1);
  // Find the left eye:
  haar_cascade.detectMultiScale(topLeftOfFace, detectedLeftEye);
  for(int i = 0; i < detectedLeftEye.size(); i++) {
    Rect eye_i = detectedLeftEye[i];
    eye_i.x += leftX;
    eye_i.y += topY;
    leftEye = Point(eye_i.x + eye_i.width/2, eye_i.y + eye_i.height/2);
  }
  // If cascade fails, try another
  if(detectedLeftEye.empty()) {
    haar_cascade.load(PATH_CASCADE_BOTHEYES);
    haar_cascade.detectMultiScale(topLeftOfFace, detectedLeftEye);
    for(int i = 0; i < detectedLeftEye.size(); i++) {
      Rect eye_i = detectedLeftEye[i];
      eye_i.x += leftX;
      eye_i.y += topY;
      leftEye = Point(eye_i.x + eye_i.width/2, eye_i.y + eye_i.height/2);
    }
  }
  //Find the right eye
  haar_cascade.load(PATH_CASCADE_RIGHTEYE);
  haar_cascade.detectMultiScale(topRightOfFace, detectedRightEye);
  for(int i = 0; i < detectedRightEye.size(); i++) {
    Rect eye_i = detectedRightEye[i];
    eye_i.x += rightX;;
    eye_i.y += topY;
    rightEye = Point(eye_i.x + eye_i.width/2, eye_i.y + eye_i.height/2);
  }
  // If cascade fails, try another
  if(detectedRightEye.empty()) {
    haar_cascade.load(PATH_CASCADE_BOTHEYES);
    haar_cascade.detectMultiScale(topLeftOfFace, detectedRightEye);
    for(int i = 0; i < detectedRightEye.size(); i++) {
      Rect eye_i = detectedRightEye[i];
      eye_i.x += leftX;
      eye_i.y += topY;
      rightEye = Point(eye_i.x + eye_i.width/2, eye_i.y + eye_i.height/2);
    }
  }
  //	Inicializacao dos kalman filters
  if(initiKalmanFilter && leftEye.x >= 0 && rightEye.x >= 0) {
    KFrightEye.statePre.at<float>(0) = rightEye.x;
    KFrightEye.statePre.at<float>(1) = rightEye.y;
    KFrightEye.statePre.at<float>(2) = 0;
    KFrightEye.statePre.at<float>(3) = 0;
    KFrightEye.transitionMatrix = (Mat_<float>(4, 4) << 1,0,0,0,   0,1,0,0,  0,0,1,0,  0,0,0,1);
    setIdentity(KFrightEye.measurementMatrix);
    setIdentity(KFrightEye.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KFrightEye.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KFrightEye.errorCovPost, Scalar::all(.1));

    KFleftEye.statePre.at<float>(0) = leftEye.x;
    KFleftEye.statePre.at<float>(1) = leftEye.y;
    KFleftEye.statePre.at<float>(2) = 0;
    KFleftEye.statePre.at<float>(3) = 0;
    KFleftEye.transitionMatrix = (Mat_<float>(4, 4) << 1,0,0,0,   0,1,0,0,  0,0,1,0,  0,0,0,1);
    setIdentity(KFleftEye.measurementMatrix);
    setIdentity(KFleftEye.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KFleftEye.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KFleftEye.errorCovPost, Scalar::all(.1));
    initiKalmanFilter = false;
  }
  //	Predicao e correcao dos kalman filter
  if(!initiKalmanFilter && leftEye.x >= 0) {
    Mat prediction = KFleftEye.predict();
    Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
    measurement(0) = leftEye.x;
    measurement(1) = leftEye.y;
      
    Point measPt(measurement(0),measurement(1));
    Mat estimated = KFleftEye.correct(measurement);
    Point statePt(estimated.at<float>(0),estimated.at<float>(1));
    leftEyePredict.x = statePt.x;
    leftEyePredict.y = statePt.y;
  }
  if(!initiKalmanFilter && rightEye.x >= 0) {
    Mat prediction = KFrightEye.predict();
    Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
    measurement(0) = rightEye.x;
    measurement(1) = rightEye.y;
      
    Point measPt(measurement(0),measurement(1));
    Mat estimated = KFrightEye.correct(measurement);
    Point statePt(estimated.at<float>(0),estimated.at<float>(1));
    rightEyePredict.x = statePt.x;
    rightEyePredict.y = statePt.y;
  }
  //if both eyes were detected
  Mat warped;
  if (leftEye.x >= 0 && rightEye.x >= 0 && ((rightEye.x - leftEye.x) > (face.cols/4))) {
    sucess = true;
    framesLost = 0;
    int desiredFaceHeight = desiredFaceWidth;
    // Get the center between the 2 eyes.
    Point2f eyesCenter = Point2f( (leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f );
    // Get the angle between the 2 eyes.
    double dy = (rightEye.y - leftEye.y);
    double dx = (rightEye.x - leftEye.x);
    double len = sqrt(dx*dx + dy*dy);
    double angle = atan2(dy, dx) * 180.0/CV_PI; // Convert from radians to degrees.

    // Hand measurements shown that the left eye center should ideally be at roughly (0.19, 0.14) of a scaled face image.
    const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);
    // Get the amount we need to scale the image to be the desired fixed size we want.
    double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * desiredFaceWidth;
    double scale = desiredLen / len;
    // Get the transformation matrix for rotating and scaling the face to the desired angle & size.
    Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
    // Shift the center of the eyes to be the desired center between the eyes.
    rot_mat.at<double>(0, 2) += desiredFaceWidth * 0.5f - eyesCenter.x;
    rot_mat.at<double>(1, 2) += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;

    // Rotate and scale and translate the image to the desired angle & size & position!
    // Note that we use 'w' for the height instead of 'h', because the input face has 1:1 aspect ratio.
    warped = Mat(desiredFaceHeight, desiredFaceWidth, CV_8U, Scalar(128)); // Clear the output image to a default grey.
    warpAffine(face, warped, rot_mat, warped.size());
    equalizeHist(warped, warped);
  }
  else {
  	framesLost++;
  	if(framesLost < MAX_FRAME_LOST && (leftEyePredict.x >= 0 || rightEyePredict.x >= 0)) {
  		if(leftEye.x < 0) {
  			leftEye.x = leftEyePredict.x;
  			leftEye.y = leftEyePredict.y;
  		}
  		if(rightEye.x < 0) {
  			rightEye.x = rightEyePredict.x;
  			rightEye.y = rightEyePredict.y;
  		}
  		if((rightEye.x - leftEye.x) > (face.cols/4)) {
	  		sucess = true;
	  		int desiredFaceHeight = desiredFaceWidth;
		    // Get the center between the 2 eyes.
		    Point2f eyesCenter = Point2f( (leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f );
		    // Get the angle between the 2 eyes.
		    double dy = (rightEye.y - leftEye.y);
		    double dx = (rightEye.x - leftEye.x);
		    double len = sqrt(dx*dx + dy*dy);
		    double angle = atan2(dy, dx) * 180.0/CV_PI; // Convert from radians to degrees.

		    // Hand measurements shown that the left eye center should ideally be at roughly (0.19, 0.14) of a scaled face image.
		    const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);
		    // Get the amount we need to scale the image to be the desired fixed size we want.
		    double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * desiredFaceWidth;
		    double scale = desiredLen / len;
		    // Get the transformation matrix for rotating and scaling the face to the desired angle & size.
		    Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
		    // Shift the center of the eyes to be the desired center between the eyes.
		    rot_mat.at<double>(0, 2) += desiredFaceWidth * 0.5f - eyesCenter.x;
		    rot_mat.at<double>(1, 2) += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;

		    // Rotate and scale and translate the image to the desired angle & size & position!
		    // Note that we use 'w' for the height instead of 'h', because the input face has 1:1 aspect ratio.
		    warped = Mat(desiredFaceHeight, desiredFaceWidth, CV_8U, Scalar(128)); // Clear the output image to a default grey.
		    warpAffine(face, warped, rot_mat, warped.size());
		    equalizeHist(warped, warped);
	    }
	    else {
	    	sucess = false;
			return face;
	    }
  	}
  	else {
		sucess = false;
		return face;
  	}
  }
  bilateralFilter(warped, faceProcessed, 0, 20.0, 2.0);
  im_height = faceProcessed.rows;
  Mat mask = Mat(faceProcessed.size(), CV_8U, Scalar(0)); // Start with an empty mask.
  Point faceRect = Point( im_height/2, cvRound(im_height * FACE_ELLIPSE_CY) );
  Size size = Size( cvRound(im_height * FACE_ELLIPSE_W), cvRound(im_height * FACE_ELLIPSE_H) );
  ellipse(mask, faceRect, size, 0, 0, 360, Scalar(255), CV_FILLED);
  // Use the mask, to remove outside pixels.
  Mat dstImg = Mat(faceProcessed.size(), CV_8U, Scalar(128)); // Clear the output image to a default gray.
  faceProcessed.copyTo(dstImg, mask);  // Copies non-masked pixels from filtered to dstImg.
  return dstImg;
}

 void read_csv(const string& filename, vector<Mat>& images, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        if(!path.empty()) {
            images.push_back(imread(path, 0));
        }
    }
}

int main(int argc, char *argv[])
{
  	//KALMAN
  	measurement.setTo(Scalar(0));
  	measurement = Mat_<float>(2,1); 
  	KFrightEye = KalmanFilter(4, 2, 0);
  	KFleftEye = KalmanFilter(4, 2, 0);
  	state = Mat_<float>(4, 1); 
  	processNoise = Mat(4, 1, CV_32F);
  	// Reconhecimento continuo
  	float P_Safe_Ultimo = 1.0, P_notSafe_Ultimo = 0.0;
  	float P_Atual_Safe, P_Atual_notSafe;
  	float P_Safe_Atual, P_notSafe_Atual;
  	float P_Safe;
  	float timeLastObs = 0, timeAtualObs = 0, tempo = 0;
    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<Mat> video;
    vector<int> labels;
    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
    //leitura dos frames
    char* path = argv[1];
    char* path2 = argv[2];
    char* visualizar = argv[3];
    STREAM *ir = createStream(path);
    STREAM *ir2 = createStream(path2);
    bool flag_ir, flag_ir2;
    flag_ir = streamNext(ir);
    //flag_ir2 = streamNext(ir2);
    struct timeval now;
    gettimeofday(&now, NULL);
    
    CascadeClassifier haar_cascade;
    haar_cascade.load(PATH_CASCADE_FACE);
    bool sucess = false;
    int login = 0;
    Point ultimaFace = Point(-1, -1);
    flag_ir2 = false;

    float PSafe1, PSafe2;
    int frame_window = 0;
    Mat faceTreino;
    double menorScore = 99999999.0;

    while(flag_ir || flag_ir2) {
        Mat frame;
        if(!flag_ir2)
    	   frame = ir->infrared;
        else
          frame = ir2->infrared;
        frame.convertTo(frame, CV_8UC3, 255, 0);
        if(visualizar[0] == '1') {
          imshow("Video", frame);
          waitKey(2);
        }
        vector< Rect_<int> > faces;
        // Find the faces in the frame:
        haar_cascade.detectMultiScale(frame, faces);
        tempo = getTickCount();
        for(int j = 0; j < faces.size(); j++) {
         	    Mat face = frame(faces[j]);
		    Rect faceRect = faces[j];
		    Point center = Point(faceRect.x + faceRect.width/2, faceRect.y + faceRect.width/2);
		    if(ultimaFace.x < 0) {
		    		ultimaFace.x = center.x;
		    		ultimaFace.y = center.y;
		    }
		    else {
		    		double res = cv::norm(ultimaFace-center);
		    		if(res < 50.0) {
		    			ultimaFace.x = center.x;
			    		ultimaFace.y = center.y;
			    		Mat faceNormalized = faceNormalize(face, FACE_SIZE, sucess);
			          if(sucess) {
                      if(visualizar[0] == '1') {
                        imshow("Normalizacao", faceNormalized);
                        waitKey(2);
                      }
			                if(login < FRAMES_LOGIN) {
			                  images.push_back(faceNormalized);
			                  labels.push_back(0);
			                  login++;
			                  if(login == FRAMES_LOGIN) 
			                    model->train(images, labels);
			                }
			                else {
			                    timeLastObs= timeAtualObs;
			                    timeAtualObs = tempo;
			                    double confidence;
			                    int prediction;
			                    model->predict(faceNormalized, prediction, confidence);
                          cout << confidence << endl;
                          /*if(confidence < menorScore) {
                            menorScore = confidence;
                            faceTreino = faceNormalized;
                          }*/
			                    //reconhecimento continuo
			                    if(timeLastObs == 0) {
			                    	P_Atual_Safe = 1 - ((1 + erf((confidence-Usafe) / (Rsafe*sqrt(2))))/2);
			                    	P_Atual_notSafe = ((1 + erf((confidence-UnotSafe) / (RnotSafe*sqrt(2))))/2);
			                        float deltaT = (timeLastObs - timeAtualObs)/getTickFrequency();
			                        float elevado = (deltaT * LN2)/K_DROP;
			                    	P_Safe_Atual = P_Atual_Safe + pow(E,elevado) * P_Safe_Ultimo;
			              		    P_notSafe_Atual = P_Atual_notSafe + pow(E,elevado) * P_notSafe_Ultimo;
			                    }
			                    else {
			                    	P_Atual_Safe = 1 - ((1 + erf((confidence-Usafe) / (Rsafe*sqrt(2))))/2);
			                    	P_Atual_notSafe = ((1 + erf((confidence-UnotSafe) / (RnotSafe*sqrt(2))))/2);
			                    	P_Safe_Ultimo = P_Safe_Atual;
			                    	P_notSafe_Ultimo = P_notSafe_Atual;
			                    	float deltaT = (timeLastObs - timeAtualObs)/getTickFrequency();
			                      float elevado = (deltaT * LN2)/K_DROP;
			                      P_Safe_Atual = P_Atual_Safe + pow(E,elevado) * P_Safe_Ultimo;
			                      P_notSafe_Atual = P_Atual_notSafe + pow(E,elevado) * P_notSafe_Ultimo;
			                    }
			                }
			            }
				 }
		    }
        }
        if(P_Safe_Atual != 0) {
          float deltaT = -(tempo - timeAtualObs)/getTickFrequency();
          float elevado = (deltaT * LN2)/K_DROP;
          P_Safe = (pow(E,elevado) * P_Safe_Atual)/(P_Safe_Atual+P_notSafe_Atual);
          //if(P_Safe > 0)
            //cout << P_Safe << endl;
        }
       /* if(frame_window == 0) {
          PSafe1 = P_Safe;
          frame_window++;
        }
        else {
          if(frame_window == WINDOWN) {
            PSafe2 = P_Safe;
            if(PSafe1 - PSafe2 > THR_QUEDA) {

            }
          }
        }*/
        
        if(!flag_ir2)
          flag_ir = streamNext(ir);
        if(!flag_ir && !flag_ir2)
          flag_ir2 = true;
        if(flag_ir2)
          flag_ir2 = streamNext(ir2);
        
    }
    return 0;
}
