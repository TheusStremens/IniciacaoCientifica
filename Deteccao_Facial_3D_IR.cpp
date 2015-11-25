#include <iostream>
#include <fstream>
#include <sstream>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/video/tracking.hpp>
#include <limits>
#include <math.h>

using namespace cv;
using namespace std;
const string PATH_CASCADE_FACE_IR = "/home/matheusm/Cascades/cascade.xml";
const string PATH_CASCADE_FACE = "/home/matheusm/Cascades/ALL_Spring2003_3D.xml";
// Image parameters
#define WIDTH 512             // Input image width
#define HEIGHT 424              // Input image height
#define SIZE 217088             // Input image size
// Calibration parameters
#define DEPTH_THRESHOLD 170         // Maximum disparity value
#define JUMP 6 //amostragem da imagem 3D para criacao da nuvem de pontos
// Detection parameters
#define X_WIDTH 1800.0            // Orthogonal projection width - in mm
#define Y_WIDTH 1600.0            // Orthogonal projection height - in mm
#define RESOLUTION 0.127272727        // Resolution in pixels per mm  0.127272727 
#define FACE_SIZE 21            // Face size - 165*RESOLUTION
#define FACE_HALF_SIZE 10         // (165*RESOLUTION)/2

bool protonect_shutdown = false;
bool firs_file = true;
bool first_proj = true;

int Taxa_Deteccao = 0, Qtd_Frames = 0;

libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);
libfreenect2::Registration* registration;

void sigint_handler(int s)
{
  protonect_shutdown = true;
}

//funcao pra gerar arquivo com nuvem de pontos
void gravaNumevemPontos(string asLinha, char* asNomeArquivo) {
	ofstream FILE;
  if(firs_file) {
    FILE.open(asNomeArquivo);
    firs_file = false;
  }
  else
	   FILE.open(asNomeArquivo, std::ios_base::app);
	FILE << asLinha;
	FILE.close();
}

// Compute rotation matrix and its inverse matrix
void computeRotationMatrix(double matrix[3][3], double imatrix[3][3], double aX, double aY, double aZ) {
  double cosX, cosY, cosZ, sinX, sinY, sinZ, d;
  cosX = cos(aX);
  cosY = cos(aY);
  cosZ = cos(aZ);
  sinX = sin(aX);
  sinY = sin(aY);
  sinZ = sin(aZ);

  matrix[0][0] = cosZ*cosY+sinZ*sinX*sinY;
  matrix[0][1] = sinZ*cosY-cosZ*sinX*sinY;
  matrix[0][2] = cosX*sinY;
  matrix[1][0] = -sinZ*cosX;
  matrix[1][1] = cosZ*cosX;
  matrix[1][2] = sinX;
  matrix[2][0] = sinZ*sinX*cosY-cosZ*sinY;
  matrix[2][1] = -cosZ*sinX*cosY-sinZ*sinY;
  matrix[2][2] = cosX*cosY;

  d = matrix[0][0]*(matrix[2][2]*matrix[1][1]-matrix[2][1]*matrix[1][2])-matrix[1][0]*(matrix[2][2]*matrix[0][1]-matrix[2][1]*matrix[0][2])+matrix[2][0]*(matrix[1][2]*matrix[0][1]-matrix[1][1]*matrix[0][2]);

  imatrix[0][0] = (matrix[2][2]*matrix[1][1]-matrix[2][1]*matrix[1][2])/d;
  imatrix[0][1] = -(matrix[2][2]*matrix[0][1]-matrix[2][1]*matrix[0][2])/d;
  imatrix[0][2] = (matrix[1][2]*matrix[0][1]-matrix[1][1]*matrix[0][2])/d;
  imatrix[1][0] = -(matrix[2][2]*matrix[1][0]-matrix[2][0]*matrix[1][2])/d;
  imatrix[1][1] = (matrix[2][2]*matrix[0][0]-matrix[2][0]*matrix[0][2])/d;
  imatrix[1][2] = -(matrix[1][2]*matrix[0][0]-matrix[1][0]*matrix[0][2])/d;
  imatrix[2][0] = (matrix[2][1]*matrix[1][0]-matrix[2][0]*matrix[1][1])/d;
  imatrix[2][1] = -(matrix[2][1]*matrix[0][0]-matrix[2][0]*matrix[0][1])/d;
  imatrix[2][2] = (matrix[1][1]*matrix[0][0]-matrix[1][0]*matrix[0][1])/d;
}

void compute_projection(IplImage *p, IplImage *p_ir, IplImage *m, CvPoint3D64f *xyz, double *intensidade, int n, double matrix[3][3], double background) {
  static int flag = 1, height, width, size, cx, cy, *li, *lj, *lc;
  int i, j, k, l, c, t;
  double d, d_ir;

  if(flag) {
    flag = 0;

    height = p->height;
    width = p->width;
    size = height*width;

    li = (int *) malloc(3*size*sizeof(int));
    lj = li+size;
    lc = lj+size;

    cx = width/2;
    cy = height/2;
  }
  // Compute projection
  cvSet(p, cvRealScalar(-DBL_MAX), NULL);
  cvSet(p_ir, cvRealScalar(0), NULL);
  cvSet(m, cvRealScalar(0), NULL);

  for(i=0; i < n; i++) {
    j = cy-cvRound(xyz[i].x*matrix[1][0]+xyz[i].y*matrix[1][1]+xyz[i].z*matrix[1][2]);
    k = cx+cvRound(xyz[i].x*matrix[0][0]+xyz[i].y*matrix[0][1]+xyz[i].z*matrix[0][2]);
    d = xyz[i].x*matrix[2][0]+xyz[i].y*matrix[2][1]+xyz[i].z*matrix[2][2];
    if(j >= 0 && k >= 0 && j < height && k < width && d > CV_IMAGE_ELEM(p, double, j, k)) {
      CV_IMAGE_ELEM(p, double, j, k) = d;
      CV_IMAGE_ELEM(p_ir, double, j, k) = intensidade[i];
      CV_IMAGE_ELEM(m, uchar, j, k) = 1;
    }
  }
  // Hole filling
  k=l=0;
  for(i=1; i < height-1; i++)
    for(j=1; j < width-1; j++)
      if(!CV_IMAGE_ELEM(m, uchar, i, j) && (CV_IMAGE_ELEM(m, uchar, i, j-1) || CV_IMAGE_ELEM(m, uchar, i, j+1) || CV_IMAGE_ELEM(m, uchar, i-1, j) || CV_IMAGE_ELEM(m, uchar, i+1, j))) {
        li[l] = i;
        lj[l] = j;
        lc[l] = 1;
        l++;
      }

  while(k < l) {
    i = li[k];
    j = lj[k];
    c = lc[k];
    if(!CV_IMAGE_ELEM(m, uchar, i, j) && i > 0 && i < height-1 && j > 0 && j < width-1 && c < FACE_HALF_SIZE) {
      CV_IMAGE_ELEM(m, uchar, i, j) = c+1;
      t = 0;
      d = 0.0f;
      d_ir = 0.0f;
      if(CV_IMAGE_ELEM(m, uchar, i, j-1) && CV_IMAGE_ELEM(m, uchar, i, j-1) <= c) {
        t++;
        d += CV_IMAGE_ELEM(p, double, i, j-1);
        d_ir += CV_IMAGE_ELEM(p_ir, double, i, j-1);
      }
      else {
        li[l] = i;
        lj[l] = j-1;
        lc[l] = c+1;
        l++;
      }
      if(CV_IMAGE_ELEM(m, uchar, i, j+1) && CV_IMAGE_ELEM(m, uchar, i, j+1) <= c) {
        t++;
        d += CV_IMAGE_ELEM(p, double, i, j+1);
        d_ir += CV_IMAGE_ELEM(p_ir, double, i, j+1);
      }
      else {
        li[l] = i;
        lj[l] = j+1;
        lc[l] = c+1;
        l++;
      }
      if(CV_IMAGE_ELEM(m, uchar, i-1, j) && CV_IMAGE_ELEM(m, uchar, i-1, j) <= c) {
        t++;
        d += CV_IMAGE_ELEM(p, double, i-1, j);
        d_ir += CV_IMAGE_ELEM(p_ir, double, i-1, j);
      }
      else {
        li[l] = i-1;
        lj[l] = j;
        lc[l] = c+1;
        l++;
      }
      if(CV_IMAGE_ELEM(m, uchar, i+1, j) && CV_IMAGE_ELEM(m, uchar, i+1, j) <= c) {
        t++;
        d += CV_IMAGE_ELEM(p, double, i+1, j);
        d_ir += CV_IMAGE_ELEM(p_ir, double, i+1, j);
      }
      else {
        li[l] = i+1;
        lj[l] = j;
        lc[l] = c+1;
        l++;
      }
      CV_IMAGE_ELEM(p, double, i, j) = d/(double)t;
      CV_IMAGE_ELEM(p_ir, double, i, j) = d_ir/(double)t;
    }
    k++;
  }
  // Final adjustments
  for(i=0; i < height; i++)
    for(j=0; j < width; j++) {
      if(CV_IMAGE_ELEM(p, double, i, j) == -DBL_MAX) {
        CV_IMAGE_ELEM(p, double, i, j) = background;
        CV_IMAGE_ELEM(p_ir, double, i, j) = 0.0;
      }
      if(CV_IMAGE_ELEM(m, uchar, i, j))
        CV_IMAGE_ELEM(m, uchar, i, j) = 1;
    }
  //salva a primeira projecao em obj
  if(first_proj) {
    firs_file = true;
    for(i=0; i < height; i++) {
      for(j=0; j < width; j++) {  
        std::ostringstream buff;
        buff << "v " << j << " " << i << " " << CV_IMAGE_ELEM(p, double, i, j) << endl;
        string linha = buff.str();  
        gravaNumevemPontos(linha, "projecao.obj");
      }
    }
    first_proj = false;
  }
}
//parametros da camera
float fx;
float fy;
float cx;
float cy;
float k1;
float k2;
float p1;
float p2;
float k3;
//converte ponto xyz para ponto na imagem
void xyz2depth(CvPoint3D64f *pt, int *i, int *j, int *s, Mat xycords) {
  /*float x, y;
  double ptX, ptY, ptZ;
  ptZ = (pt->z-80)*2.1;
  ptX = pt->x;
  ptY = pt->y;
  x = (fx * ptX)/ptZ + cx;
  y = (fy * ptY)/ptZ + cy;
  *s = 65.0;
  int p;
  for(p = 0; p < 217088; p+=JUMP) {
    cv::Vec2f xy = xycords.at<cv::Vec2f>(0, p);
    if(fabs(x - xy[1]) < 0.9 && fabs(y - xy[0]) < 0.9)
      break;
  }
  if(p < 512) {
    *i = 0;
    *j = p;
  }
  else {
    *i = p / 512;
    *j = p % 512;
  }*/
    /*x = (c + 0.5 - cx) * fx * depth_val;
    y = (r + 0.5 - cy) * fy * depth_val;
    z = depth_val;*/
    double ptX, ptY, ptZ;
    ptX = pt->x/100.0f;
    ptY = -pt->y/100.0f;
    ptZ = -pt->z/100.0f;
    const float FX(1/fx), FY(1/fy);
    *j = -0.5*ptZ + cx + (ptX/(FX*ptZ));
    *i = -0.5*ptZ + cy + (ptY/(FY*ptZ));
    *s = 65;
}
//Funcao principal de detecao facial
void face_detection(Mat depth_image, Mat ir_image, int minX, int maxX, int minY, int maxY, int minZ, int maxZ, Mat xycords, vector<Vec4i> &faces) {
  
  static CvPoint3D64f *xyz, *list, *clist;
  static double *intensidade;
  CvPoint3D64f avg;
  
  float* ptr = (float*) (depth_image.data);
  float* ir_ptr = (float*) (ir_image.data);
  static CvHaarClassifierCascade *face_cascade;
  //float x = 0.0f, y = 0.0f;
  uint pixel_count = depth_image.rows * depth_image.cols;
  double menorX = 999999.0, menorY = 999999.0, menor = 999999.0;
  double maiorX = 0.0, maiorY = 0.0, maior = 0.0;
  static IplImage *p, *p_ir, *pColorida, *pColorida3D, *m, *sum, *sqsum, *tiltedsum, *msum, *sumint, *tiltedsumint;;
  static int width, height, CX, CY, flag = 1;
  double matrix[3][3], imatrix[3][3], background, X, Y, Z;
  
  if(flag) {
    xyz = (CvPoint3D64f *) malloc(SIZE*sizeof(CvPoint3D64f));
    intensidade = (double *) malloc(SIZE*sizeof(double));
  }

  int n = 0;
  //criacao da nuvem de pontos 3D xyz
  float x, y, z, rgb, dpt;
  const float bad_point = std::numeric_limits<float>::quiet_NaN();
  for (int r=0; r<424; r+=JUMP)
  for (int c=0; c<512; c+=JUMP) {
    dpt = depth_image.at<float>(r,c)*1000.0f;
    if(dpt < DEPTH_THRESHOLD) {
      registration->getPointXYZRGB(&undistorted, &registered, r,c, x,y,z,rgb);
      if(x == x || y==y || z==z) {
        xyz[n].z = -z * 100.0f;
        xyz[n].y = -y * 100.0f;
        xyz[n].x = x * 100.0f;
        if(xyz[n].z < menor)
          menor = xyz[n].z;
        intensidade[n] = ir_image.at<float>(r,c)*1000.0f;
        if(flag) {
            std::ostringstream buff;
            buff << "v " << xyz[n].x << " " << xyz[n].y << " " << xyz[n].z << endl;
            string linha = buff.str();  
            gravaNumevemPontos(linha, "nuvemPontos.obj");
        }
        n++;
      }
    }
  }
  /*for (uint i = 0; i < pixel_count; i+=JUMP)
  {
      cv::Vec2f xy = xycords.at<cv::Vec2f>(0, i);
      x = xy[1]; y = xy[0];
      double dpt = -(static_cast<float>(*ptr)) * (1000.0f); // Converte metros pra mm
      double ir_value = (static_cast<float>(*ir_ptr)); // Converte metros pra mm
      if(-dpt < DEPTH_THRESHOLD) {
        xyz[n].z = dpt;
        xyz[n].z /= 2.1; //ajuste
        xyz[n].x = (x - cx) * xyz[n].z / fx;
        xyz[n].y = (y - cy) * xyz[n].z / fy;
        xyz[n].z += 80; //ajuste
        intensidade[n] = ir_value * (1000.0f); //ajuste para que os valores do IR não fiquem tão pequenos
        if(xyz[n].z < menor)
          menor = xyz[n].z;
        //salva a primeira nuvem de pontos 3D
        if(flag) {
          std::ostringstream buff;
          buff << "v " << xyz[n].x << " " << xyz[n].y << " " << xyz[n].z << endl;
          string linha = buff.str();  
          gravaNumevemPontos(linha, "nuvemPontos.obj");
        }
        n++;
      }
      ptr+=JUMP;
      ir_ptr+=JUMP;
  }*/
  background = menor;
  ///iniciazliacao de variaveis da projecao
  if(flag) {
      flag = 0;

      width = (int)(X_WIDTH*RESOLUTION);
      height = (int)(X_WIDTH*RESOLUTION);
      CX = width/2;
      CY = height/2;

      p = cvCreateImage(cvSize(width, height), IPL_DEPTH_64F, 1);
      p_ir = cvCreateImage(cvSize(width, height), IPL_DEPTH_64F, 1);
      pColorida = cvCreateImage(cvSize(width, height), IPL_DEPTH_64F, 1);
      pColorida3D = cvCreateImage(cvSize(width, height), IPL_DEPTH_64F, 1);
      m = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

      face_cascade = (CvHaarClassifierCascade *) cvLoad("/home/matheusm/Cascades/ALL_Spring2003_3D.xml", 0, 0, 0);
      sum = cvCreateImage(cvSize(width+1, height+1), IPL_DEPTH_64F, 1);
      sqsum = cvCreateImage(cvSize(width+1, height+1), IPL_DEPTH_64F, 1);
      tiltedsum = cvCreateImage(cvSize(width+1, height+1), IPL_DEPTH_64F, 1);
      sumint = cvCreateImage(cvSize(width+1, height+1), IPL_DEPTH_32S, 1);
      tiltedsumint = cvCreateImage(cvSize(width+1, height+1), IPL_DEPTH_32S, 1);
      msum = cvCreateImage(cvSize(width+1, height+1), IPL_DEPTH_32S, 1);

      list = (CvPoint3D64f *) malloc(2000*sizeof(CvPoint3D64f));
      clist = list+1000;
  }

  int i, j, k = 0, l, aX, aY, aZ;
  //loop de detecao facial
  Mat colored, colored3D;
  cv::Mat projecao_colorida_3d;
  for(aX=minX, k=0; aX <= maxX; aX += 10) {
    for(aY=minY; aY <= maxY; aY += 10) {
      for(aZ=minZ; aZ <= maxZ; aZ += 10) {
        if(aX+aY+aZ > 30)
          continue;
        computeRotationMatrix(matrix, imatrix, aX*0.017453293, aY*0.017453293, aZ*0.017453293);
        compute_projection(p, p_ir, m, xyz, intensidade, n, matrix, background);
        //exibicao da imagem de projecao IR
        menor = 999999.0; 
        for(i = 0; i < width; i++) {
          for(j = 0; j < height; j++) {
            double x = CV_IMAGE_ELEM(p_ir, double, i, j);
            if(x > maior)
              maior = x;
            if(x < menor && x != 0)
              menor = x;
          }
        }
        double a, b;
        a = 255/(maior-menor);
        b = 1 - (menor * a);
        for(i = 0; i < width; i++) {
          for(j = 0; j < height; j++) {
            x = CV_IMAGE_ELEM(p_ir, double, i, j);
            if(x != 0)
              CV_IMAGE_ELEM(pColorida, double, i, j) = (x * a) + b;
          }
        }
        Mat projecao= cv::cvarrToMat(pColorida); 
        Mat1b projecao_colorida(projecao.rows, projecao.cols);
        for(i = 0; i < projecao.rows; i++)
          for(j = 0; j < projecao.cols; j++) 
            projecao_colorida.at<uint8_t>(i, j) = projecao.at<double>(i, j);
        colored = projecao_colorida;
        //exibicao da imagem de projecao 3D colorida
        menor = 999999.0; 
        maior = 0.0;
        for(i = 0; i < width; i++) {
          for(j = 0; j < height; j++) {
            double x = CV_IMAGE_ELEM(p, double, i, j);
            if(x > maior)
              maior = x;
            if(x < menor && x != 0)
              menor = x;
          }
        }
        a = 255/(maior-menor);
        b = 1 - (menor * a);
        for(i = 0; i < width; i++) {
          for(j = 0; j < height; j++) {
            x = CV_IMAGE_ELEM(p, double, i, j);
            if(x != 0)
              CV_IMAGE_ELEM(pColorida3D, double, i, j) = (x * a) + b;
          }
        }
        Mat projecao3D = cv::cvarrToMat(pColorida3D); 
        Mat1b projecao_colorida3D(projecao3D.rows, projecao3D.cols);
        for(i = 0; i < projecao3D.rows; i++)
          for(j = 0; j < projecao3D.cols; j++) 
            projecao_colorida3D.at<uint8_t>(i, j) = projecao3D.at<double>(i, j);
        applyColorMap(projecao_colorida3D, colored3D, COLORMAP_JET);
        //detecao facial 3D na imagem de projecao 
        cvIntegral(p, sum, sqsum, tiltedsum);
        cvIntegral(m, msum, NULL, NULL);
        
        for(i=0; i < height+1; i++)
          for(j=0; j < width+1; j++) {
            CV_IMAGE_ELEM(sumint, int, i, j) = CV_IMAGE_ELEM(sum, double, i, j);
            CV_IMAGE_ELEM(tiltedsumint, int, i, j) = CV_IMAGE_ELEM(tiltedsum, double, i, j);
          }

        cvSetImagesForHaarClassifierCascade(face_cascade, sumint, sqsum, tiltedsumint, 1.0);

        for(i=0; i < height-20; i++)
          for(j=0; j < width-20; j++)
            if(CV_IMAGE_ELEM(msum, int, i+FACE_SIZE, j+FACE_SIZE)-CV_IMAGE_ELEM(msum, int, i, j+FACE_SIZE)-CV_IMAGE_ELEM(msum, int, i+FACE_SIZE, j)+CV_IMAGE_ELEM(msum, int, i, j) == 441)
              if(cvRunHaarClassifierCascade(face_cascade, cvPoint(j,i), 0) > 0) {
                rectangle(colored3D, Point(j, i), Point(j+21, i+21), CV_RGB(0,255,0));
                X = (j+FACE_HALF_SIZE-CX)/RESOLUTION;
                Y = (CY-i-FACE_HALF_SIZE)/RESOLUTION;
                Z = (CV_IMAGE_ELEM(sum, double, i+FACE_HALF_SIZE+6, j+FACE_HALF_SIZE+6)-CV_IMAGE_ELEM(sum, double, i+FACE_HALF_SIZE-5, j+FACE_HALF_SIZE+6)-CV_IMAGE_ELEM(sum, double, i+FACE_HALF_SIZE+6, j+FACE_HALF_SIZE-5)+CV_IMAGE_ELEM(sum, double, i+FACE_HALF_SIZE-5, j+FACE_HALF_SIZE-5))/121.0/RESOLUTION;
                
                list[k].x = X*imatrix[0][0]+Y*imatrix[0][1]+Z*imatrix[0][2];
                list[k].y = X*imatrix[1][0]+Y*imatrix[1][1]+Z*imatrix[1][2];
                list[k].z = X*imatrix[2][0]+Y*imatrix[2][1]+Z*imatrix[2][2];
                
                k++;
        }
      }
    }
  }
  //detecao facial IR caso a deteccao 3D falhe
  vector< Rect_<int> > faces_ir;
  if(k == 0) {
    CascadeClassifier haar_cascade;
    haar_cascade.load(PATH_CASCADE_FACE_IR);
    haar_cascade.detectMultiScale(colored, faces_ir, 1.1, 3, 0, Size(19,19), Size(23,23));
    for(int i = 0; i < faces_ir.size(); i++) {
      Rect face_i = faces_ir[i];
      rectangle(colored, face_i, CV_RGB(0, 0, 255), 1);
    }
  }
  //exibição das imagens de projecao
  cv::imshow("Imagem de Projecao IR", colored);
  cv::imshow("Imagem de Projecao 3D", colored3D);
  // Merge multiple detections
  Vec4i tmp;
  while(k > 0) {

    avg.x = clist[0].x = list[0].x;
    avg.y = clist[0].y = list[0].y;
    avg.z = clist[0].z = list[0].z;
    list[0].x = DBL_MAX;
    
    j=1;
    
    for(l=0; l < j; l++)
      for(i=1; i < k; i++)
        if(list[i].x != DBL_MAX) {
          X = sqrt(pow(list[i].x-clist[l].x, 2.0)+pow(list[i].y-clist[l].y, 2.0)+pow(list[i].z-clist[l].z, 2.0));
          if(X < 50.0) {
            
            avg.x += clist[j].x = list[i].x;
            avg.y += clist[j].y = list[i].y;
            avg.z += clist[j].z = list[i].z;
            list[i].x = DBL_MAX;
            
            j++;
          }
        }

    avg.x /= j;
    avg.y /= j;
    avg.z /= j;
    
    xyz2depth(&avg, &tmp[1], &tmp[0], &tmp[2], xycords);
    
    tmp[3] = j;
    faces.push_back(tmp);

    j=0;
    
    for(i=1; i < k; i++)
      if(list[i].x != DBL_MAX) {
        list[j].x = list[i].x;
        list[j].y = list[i].y;
        list[j].z = list[i].z;
        j++;
      }
    k=j;
    
  }
}

void face_detection(Mat depth, Mat ir, Mat xycords, vector<Vec4i> &faces) {
  face_detection(depth, ir, 0, 30, -20, 20, 0, 0, xycords, faces);
}

void frontal_face_detection(Mat depth, Mat ir, Mat xycords, vector<Vec4i> &faces) {
  face_detection(depth, ir, 0, 0, 0, 0, 0, 0, xycords, faces);
}



int main(int argc, char *argv[])
{
  std::string program_path(argv[0]);
  size_t executable_name_idx = program_path.rfind("Protonect");
  std::string binpath = "/";
  if(executable_name_idx != std::string::npos)
    binpath = program_path.substr(0, executable_name_idx);
  libfreenect2::Freenect2 freenect2;
  libfreenect2::Freenect2Device *dev = 0;
  libfreenect2::PacketPipeline *pipeline = 0;
  if(freenect2.enumerateDevices() == 0)
  {
    std::cout << "no device connected!" << std::endl;
    return -1;
  }
  std::string serial = freenect2.getDefaultDeviceSerialNumber();
  bool viewer_enabled = true;
  for(int argI = 1; argI < argc; ++argI)
  {
    const std::string arg(argv[argI]);
    if(arg == "cpu")
    {
      if(!pipeline)
        pipeline = new libfreenect2::CpuPacketPipeline();
    }
    else if(arg == "gl")
    {
  #ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenGLPacketPipeline();
  #else
      std::cout << "OpenGL pipeline is not supported!" << std::endl;
  #endif
    }
    else if(arg == "cl")
    {
  #ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenCLPacketPipeline();
  #else
      std::cout << "OpenCL pipeline is not supported!" << std::endl;
  #endif
    }
    else if(arg.find_first_not_of("0123456789") == std::string::npos) //check if parameter could be a serial number
    {
      serial = arg;
    }
    else if(arg == "-noviewer")
    {
      viewer_enabled = false;
    }
    else
    {
      std::cout << "Unknown argument: " << arg << std::endl;
    }
  }
  if(pipeline)
  {
    dev = freenect2.openDevice(serial, pipeline);
  }
  else
  {
    dev = freenect2.openDevice(serial);
  }
  if(dev == 0)
  {
    std::cout << "failure opening device!" << std::endl;
    return -1;
  }
  signal(SIGINT,sigint_handler);
  protonect_shutdown = false;
  libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
  libfreenect2::FrameMap frames;
  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);
  dev->start();
  std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
  std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;
  registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
  //inicializacao dos parametros da camera
  fx = dev->getIrCameraParams().fx;
  fy = dev->getIrCameraParams().fy;
  cx = dev->getIrCameraParams().cx;
  cy = dev->getIrCameraParams().cy;
  k1 = dev->getIrCameraParams().k1;
  k2 = dev->getIrCameraParams().k2;
  p1 = dev->getIrCameraParams().p1;
  p2 = dev->getIrCameraParams().p2;
  k3 = dev->getIrCameraParams().k3;

  int width = 512;
  int height = 424;
  //criacao do sistema de coordenadas distorcidas
  cv::Mat cv_img_cords = cv::Mat(1, width*height, CV_32FC2);
  Mat cv_img_corrected_cords;
  for (int r = 0; r < height; ++r) {
      for (int c = 0; c < width; ++c) {
          cv_img_cords.at<cv::Vec2f>(0, r*width + c) = cv::Vec2f((float)r, (float)c);
      }
  }
  cv::Mat k = cv::Mat::eye(3, 3, CV_32F);
  k.at<float>(0,0) = fx;
  k.at<float>(1,1) = fy;
  k.at<float>(0,2) = cx;
  k.at<float>(1,2) = cy;
  cv::Mat dist_coeffs = cv::Mat::zeros(1, 8, CV_32F);
  dist_coeffs.at<float>(0,0) = k1;
  dist_coeffs.at<float>(0,1) = k2;
  dist_coeffs.at<float>(0,2) = p1;
  dist_coeffs.at<float>(0,3) = p2;
  dist_coeffs.at<float>(0,4) = k3;
  cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(k, dist_coeffs, cv::Size2i(height,width), 0.0);
  cv::undistortPoints(cv_img_cords, cv_img_corrected_cords, k, dist_coeffs, cv::noArray(), new_camera_matrix);
  Mat xycords = cv_img_corrected_cords;
  //loop kinect  
  while(!protonect_shutdown)
  {
    listener.waitForNewFrame(frames);
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
    libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *color = frames[libfreenect2::Frame::Color];
    Mat depth_image = cv::Mat(depth->height, depth->width, CV_32FC1, depth->data) / 4500.0f;
    Mat ir_image = cv::Mat(ir->height, ir->width, CV_32FC1, ir->data) / 80000.0f;
    Mat color_image = cv::Mat(color->height, color->width, CV_8UC4, color->data);
    cv::resize(color_image, color_image, Size(color_image.cols/2, color_image.rows/2), 1.0, 1.0, INTER_CUBIC);
    vector<Vec4i> faces;

    registration->apply(color, depth, &undistorted, &registered);
    Mat registered_image = cv::Mat(registered.height, registered.width, CV_8UC4, registered.data);
    //deteca as faces
    face_detection(depth_image, ir_image, xycords, faces);
    //gera imagem colorida de profundidade
    double min;
    double max;
    cv::minMaxIdx(depth_image, &min, &max);
    cv::Mat auxiliar;
    depth_image.convertTo(auxiliar,CV_8UC1, 255 / (max-min), -min); 
    cv::Mat depth_colorida;
    applyColorMap(auxiliar, depth_colorida, cv::COLORMAP_JET);
    //marcar as faces na imagem colorida
    for(int i=0; i < faces.size(); i++)
      rectangle(registered_image, Point(faces[i][0]-faces[i][2],faces[i][1]-faces[i][2]), Point(faces[i][0]+faces[i][2],faces[i][1]+faces[i][2]), CV_RGB(0,255,0), 2, 8, 0);
    imshow("Deteccao", registered_image);
    int key = cv::waitKey(1);
    protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27)); // shutdown on escape
    listener.release(frames);
  }
  dev->stop();
  dev->close();
  return 0;
}
