#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>
using namespace std;
using namespace cv;



struct passIO{
	int id;
	bool IO;
	int nach;
	int gde;
	int gdeDo;
}

//GLOBAL
int k=0;
int pass=15;
int in=0,out=0;
const int dd=3;
Mat ures(Size(640,480),CV_32FC1);
Mat ures2(Size(640,480),CV_32FC1);
Mat res,frameMOT;
Mat shabPyr[dd];
Mat tri(Size(640,480),CV_8UC3);
bool flag;
bool outin;
Point pt;
int t=0,q=0,parl=0;
int d;
bool isWrite=false;

Mat m_openingKernel2(7,7,CV_8UC1);


void detect(Mat img) {
  string cascadeName1 = "cascade2.xml";
  string cascadeHead = "cascade1.xml";
  CascadeClassifier detectorBody;
  bool loaded1 = detectorBody.load(cascadeName1);
  CascadeClassifier detectorHead; 
  bool loaded2 = detectorHead.load(cascadeHead);
  Mat original;
  img.copyTo(original);
  vector<Rect> human;
  vector<Rect> head;
  //cv::cvtColor(img, img, CV_BGR2GRAY);
  cv::equalizeHist(img, img);
  //detectorBody.detectMultiScale(img, human, 1.04, 5, 0 | 1, Size(150, 50), Size(450,150));
  detectorHead.detectMultiScale(img, head, 1.1, 5, 0 | 1, Size(50, 40), Size(125, 100));
 // if (human.size() > 0) { 
	//  for (int gg = 0; gg < human.size(); gg++) { 
	//	  rectangle(original, human[gg].tl(), human[gg].br(), Scalar(0, 0, 255), 2, 8, 0); } } 
  if (head.size() > 0) { 
	  for (int gg = 0; gg < head.size(); gg++) { 
		  rectangle(original, head[gg].tl(), head[gg].br(), Scalar(255, 0, 0), 2, 8, 0); } }
  imshow("DetectHum", original);
}

void InOut(int x,int y,int w,int h){
	Point temp;
	temp.x=x+(w/2);
	temp.y=y+(h/2);
	cv::circle(tri,temp,3,Scalar(255,100,50),1);
	if (k==0) d=temp.x;

	if (temp.x<pt.x) {
		if (!outin) k=0;
		cv::line(tri,pt,temp,Scalar(255,100,50),2,8,0);
		k++;
		if(k > 10) 
			cout <<"pass: " << pass << "temp.x = " << temp.x << "outin: " << outin << "d: " << d << endl;
		if ((k>10)&&(temp.x<250)&&(outin)&&(d > 250)) {
			pass--;
			out++;
			cout<<"pass"<<pass<<endl;
			tri.setTo(0);
			//d=temp.x;
			k=0;
		}
		outin=true;
	} else {
		if (outin) k=0;
		cv::line(tri,pt,temp,Scalar(50,100,255),2,8,0);
		k++;
		if ((k>10)&&(temp.x>330)&&(!outin)&&(d<330)) {
			pass++;
			in++;
			cout<<"pass"<<pass<<endl;
			tri.setTo(0);
			k=0;
		}
		outin=false;
	}
	cout<<"k= "<<k<<endl;
	pt=temp;
	flag=true;

}

void motion(Mat img){
	img.copyTo(frameMOT);
	cv::cvtColor(img,img,COLOR_RGB2GRAY);
	cv::GaussianBlur(img,img,Size(75,75),-1);
	Mat tmp;
	absdiff(img,res,tmp);
	cv::threshold(tmp,tmp,5,255,THRESH_BINARY);
	//cv::morphologyEx(tmp,tmp,cv::MORPH_CLOSE, cv::Mat());
	//cv::morphologyEx(tmp,tmp,cv::MORPH_OPEN,m_openingKernel2,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
	cv::updateMotionHistory(tmp,ures,t,7.);
	ures.copyTo(ures2);
	//Mat tmp;
	//absdiff(ures,ures2,tmp);
	img.copyTo(frameMOT);
	dilate(ures,ures2,m_openingKernel2,Point(-1,-1),3);
	std::vector<cv::Rect> targets;  
	cv::segmentMotion(ures2,tmp,targets,t,7.0);
	
	if (!targets.empty())  {
		for(int j =  0; j<targets.size();j++) if ((targets[j].height+targets[j].width)>500)
		{
			//detect(img(targets[j]));//.copyTo(img1(Rect(1550,25,320,240)));
			rectangle(frameMOT, targets[j].tl(), targets[j].br(), Scalar(0, 0, 255), 2, 8, 0); 
			InOut(targets[j].x,targets[j].y,targets[j].width,targets[j].height);
		}
	} else {
		flag=false;
		k=0;
		
	}
	imshow("MotionDetector",frameMOT);
	//tablo=tri(Rect(500,350,100,100));
	//tablo.zeros(640,480,CV_8UC3);
	string tr="In:"+std::to_string((long double)in)+"  Out:"+std::to_string((long double)out)+"  Into"+std::to_string((long double)pass);
	cv::putText(tri,tr,Point(150,450),16, 1.0, CV_RGB(0,0,255), 2.0);
	imshow("DetectHum",tri);
	img.copyTo(res);
	t++;
}

void parliament(Mat img){
	Mat c;
	img.copyTo(c);
	cvtColor(img,img,COLOR_RGB2GRAY);
	double minn=9999999999,sx,sy;
	cv::Point minnloc;
	int kof=1;
	
	for (int i =0; i<dd; i++){
		for (int j=0;j<5;j++) {
			cv::Point center=(shabPyr[i].cols/2,shabPyr[i].rows/2);
			Mat M=cv::getRotationMatrix2D(center,10*j,1.0);
			Mat rot;
			cv::warpAffine(shabPyr[i],rot,M,cv::Size(shabPyr[i].cols,shabPyr[i].rows));
			//imshow("12",rot);
			Mat ures2;
		
			cv::matchTemplate(img,rot,ures2,TM_SQDIFF);
			double max,min;
			cv::Point minloc,maxloc;
			cv::minMaxLoc(ures2,&min,&max,&minloc,&maxloc);
			if (minn>min) {
				minn=min; 
				minnloc.x=minloc.x; 
				minnloc.y=minloc.y;
				sx=rot.cols; 
				sy=rot.rows;
				//cout<<min<<endl;
				cv::rectangle(c,cv::Point(minnloc.x,minnloc.y),cv::Point(minnloc.x+sx-1,minnloc.y+sy-1),Scalar(255-50*i, 0, 0), 2, 8, 0);
			}
		}
	//imshow("shab",img);
	//imshow("Parliament2",ures2);
		//cvtColor(img,img,COLOR_GRAY2RGB);
	//cout<<max-min<<endl;
		kof=kof*6.5;
	}
	
		cv::rectangle(c,cv::Point(minnloc.x,minnloc.y),cv::Point(minnloc.x+sx-1,minnloc.y+sy-1),Scalar(255, 0, 255), 2, 8, 0);
	
	//ures2.type(CV_8UC1);
	imshow("Parliament",c);

	
}

void screenT(Mat img1,Mat img2){
	cv::resize(img1,img1,Size(1920,1080));
	cv::resize(img2,img2,Size(320,240));
	//Mat tmp=img1(Rect(670,10,320,240));
	img2.copyTo(img1(Rect(1550,25,320,240)));
	//cv::line(img1,Point(512,384-kw-kl),Point(512,384-kw),Scalar(255,255,255),3,8,0);
	//cv::line(img1,Point(512,384+kw),Point(512,384+kw+kl),Scalar(255,255,255),3,8,0);
	//cv::line(img1,Point(512-kw-kl,384),Point(512-kw,384),Scalar(255,255,255),3,8,0);
	//cv::line(img1,Point(512+kw,384),Point(512+kw+kl,384),Scalar(255,255,255),3,8,0);
	imshow("ROOTERIABLE",img1);
}


 int main(){
	 cv::namedWindow("DetectHum",CV_WINDOW_NORMAL);
	 cv::namedWindow("MotionDetector",CV_WINDOW_NORMAL);
	 const string NAME = "1.avi"; 
	 cv::VideoCapture cap(0);
	 //cv::namedWindow("tmp",CV_WINDOW_NORMAL);
	 //tri.zeros();
	 Mat frame;
	 flag=false;
	 pt.x=320;
	 pt.y=240;
	 Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH), (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	 int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));// Open the output
	 //int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
	 VideoWriter outputVideo("11.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(640, 480), true);
	 if(!outputVideo.isOpened())
		 cout << "err" << endl;
	 cv::line(tri,Point(330,1),Point(330,480),Scalar(0,255,0),3);
	 cv::line(tri,Point(250,1),Point(250,480),Scalar(0,0,255),3);
	
	 if (cap.isOpened())
	 for (int i=0;i<20;i++)
	 {
		 cap>>frame;
		 cap>>res;

	 }
	 cv::cvtColor(res,res,COLOR_RGB2GRAY);
	 if (cap.isOpened())
		 while (true)
		 {
			 cap>>frame;
			 //cv::cvtColor(res,res,COLOR_RGB2GRAY);
			 //imshow("tmp",frame);
			 //detect(frame);
			 motion(frame);
			 cv::addWeighted(frame,0.2,tri,1.2,1.5,frame);
			 imshow("tmp",frame);
			 if (isWrite)
				outputVideo << frame;
			 int l=waitKey(10);
			 cout<<l<<endl;
			 if (l==119) isWrite=!isWrite;
			 if (l==113) cv::imwrite("1.bmp",frame);
			 if (l==27) break;
			 
		 }
	return 0;

 }