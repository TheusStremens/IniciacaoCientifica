#include <iostream>
#include <stdio.h>
#include <zlib.h>
//#include <lzf.h>
#include "liblzf/lzf.h"
#include "liblzf/lzf_d.c"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <linux/input.h>

using namespace cv;
using namespace std;

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
//					r->infrared.create(r->height, r->width, CV_8U);
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
//			s->infrared.release();
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

/* Magic function */
int main(int argc, char **argv) {
	STREAM *dir = createStream("../build/depth.log");
	char* path = argv[1];
	STREAM *ir = createStream(path);
	STREAM *color = createStream("../build/color.log");
	DATA *keyboard = createData("../build/keyboard.log");
	DATA *mouse = createData("../build/mouse.log");
	bool flag_dir, flag_ir, flag_color, flag_keyboard, flag_mouse;

	flag_dir = streamNext(dir);
	flag_ir = streamNext(ir);
	flag_color = streamNext(color);
	flag_keyboard = dataNext(keyboard);
	flag_mouse = dataNext(mouse);
	struct timeval now;
	gettimeofday(&now, NULL);

	while(flag_dir || flag_ir || flag_color || flag_keyboard || flag_mouse) {
		/* Who comes first? */
		struct timeval t[4];
		t[0] = (flag_dir ? dir->t : now);
		t[1] = (flag_color ? color->t : now);
		t[2] = (flag_ir ? ir->t : now);
		t[3] = (flag_keyboard ? keyboard->ev.time : now);
		t[4] = (flag_mouse ? mouse->ev.time : now);
		int min=0;
		for(int i=1; i < 4; i++)
			if(t[i].tv_sec < t[min].tv_sec || (t[i].tv_sec == t[min].tv_sec && t[i].tv_usec < t[min].tv_usec))
				min = i;

		switch(min) {
			case 0:
				imshow("depth", dir->depth);
//				imshow("infrared", dir->infrared);
				waitKey(10);
				flag_dir = streamNext(dir);
				break;
			case 1:
				imshow("color", color->color);
				waitKey(10);
				flag_color = streamNext(color);
				break;
			case 2:
				imshow("ir", ir->infrared);
				waitKey(10);
				flag_ir = streamNext(ir);
				break;
			case 3:
				printf("KEYBOARD %ld.%06ld %d %d %d\n", keyboard->ev.time.tv_sec, keyboard->ev.time.tv_usec, keyboard->ev.type, keyboard->ev.code, keyboard->ev.value);
				flag_keyboard = dataNext(keyboard);
				break;
			case 4:
				printf("MOUSE %ld.%06ld %d %d %d\n", mouse->ev.time.tv_sec, mouse->ev.time.tv_usec, mouse->ev.type, mouse->ev.code, mouse->ev.value);
				flag_mouse = dataNext(mouse);
				break;
		}
	}
	releaseStream(ir);
	releaseStream(dir);
	releaseStream(color);
}

