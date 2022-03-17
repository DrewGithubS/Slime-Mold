#ifndef ANIMATION_H
#define ANIMATION_H

#include <cstdint>

struct Particle;

class Animation {
private:
	uint32_t width;
	uint32_t height;
	uint32_t particleCount;
	Particle * particles;
	uint32_t imageSize;
	uint32_t * d_image;
	uint32_t * image;
	uint32_t blockCountGPU;
	uint64_t * h_rand;
	uint64_t * d_rand;
	int64_t * d_occupied;
	uint32_t * color;
public:
	Animation(uint32_t widthIn, uint32_t heightIn, uint32_t particlesIn);
	void init();
	void nextFrame();
	void exit();
	void * getImage();
};

#endif