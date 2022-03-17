#include <cstdint>
#include <iostream>

#include "Animation.h"
#include "GPUFunctions.h"

const float UINT64MAX = (float) ((~(uint64_t) 0));
const uint32_t THREADSPERBLOCK = 1024;
const uint32_t TRAILCOLOR = 0x00ff00ff;
const uint32_t COLORFADERATE = 0x00060006;
const float STARTINGRADIUS = 300;
const float PI = 3.14159265359;
const float SPEED = 1;
const float SCANOFFSET = 40;
const int32_t SCANRADIUS = 10;
const float SCANANGLE = PI/4;
const float MAXTURN = SCANANGLE/40;
const uint32_t DELTAANGLES = 3;
__device__ const uint64_t WEIGHTS[DELTAANGLES] = {3, 2, 3};

struct Particle {
	float posX;
	float posY;
	float angle;
};


__device__ float genRand(uint64_t * rand) {
	(*rand) *= 124513452345;
	(*rand) *= 5234958729898;
	(*rand) >>= 16;
	(*rand) *= 52384570923;
	(*rand) >>= 10;
	(*rand) *= 1098275091222;
	(*rand) *= 9234589723443;
	return ((float) (*rand)) / UINT64MAX;
}
__global__ void randomizeValues(Particle * particles, uint32_t particleCount, uint32_t width, uint32_t height, uint64_t * rand) {
	uint32_t index = blockIdx.x *blockDim.x + threadIdx.x;

	if(index < particleCount) {
		float radius = genRand(&rand[index]) * STARTINGRADIUS;
		float angle  = genRand(&rand[index]) * 2 * PI;
		particles[index].posX = cos(angle) * radius + width/2;// + (genRand(&rand[index])-0.5);
		particles[index].posY = sin(angle) * radius + height/2;// + (genRand(&rand[index])-0.5);
		particles[index].angle = -angle - PI/4;// * (rand[index] % 2 ? -1 : 1);
	}
}

__global__ void nextFrameGPU(Particle * particles, uint32_t particleCount, uint32_t width, uint32_t height, uint32_t * image, uint64_t * rand) {
	uint32_t index = blockIdx.x *blockDim.x + threadIdx.x;

	if(index < particleCount) {
		float velX = cos(particles[index].angle) * SPEED;
		float velY = sin(particles[index].angle) * SPEED;

		particles[index].posX += velX;
		particles[index].posY += velY;

		if((int32_t) particles[index].posX < 0) {
			particles[index].posX = 0;
			particles[index].angle = genRand(&rand[index]) * PI - PI/2;
		}

		if((int32_t) particles[index].posX >= width) {
			particles[index].posX = (float) (width - 1);
			particles[index].angle = (genRand(&rand[index]) * PI + PI/2);
		}

		if((int32_t) particles[index].posY < 0) {
			particles[index].posY = 0;
			particles[index].angle = genRand(&rand[index]) * PI;
		}
		
		if((int32_t) particles[index].posY >= height) {
			particles[index].posY = (float) (height - 1);
			particles[index].angle = -genRand(&rand[index]) * PI;
		}

		image[((uint32_t) (particles[index].posY)) * width + ((uint32_t) particles[index].posX)] = TRAILCOLOR;

	}
}

// __global__ void parallelParticleScan(uint32_t passedIndex, uint32_t samples, uint32_t xDim, int * sum, Particle * particles, uint32_t width, uint32_t height, uint32_t * image) {
// 	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
// 	uint32_t angle = index / samples;

// 	if(angle < DELTAANGLES) {
// 		float deltaAngle[DELTAANGLES] = {0, -SCANANGLE, SCANANGLE};

// 		uint32_t coordOffsets = index % samples;

// 		int32_t y = coordOffsets / xDim - xDim/2;
// 		int32_t x = coordOffsets % xDim - xDim/2;
// 		int32_t finalY = y + (uint32_t) (particles[passedIndex].posX + SCANOFFSET * cos(particles[passedIndex].angle + deltaAngle[angle]));
// 		int32_t finalX = x + (uint32_t) (particles[passedIndex].posY + SCANOFFSET * sin(particles[passedIndex].angle + deltaAngle[angle]));
// 		if(finalX >= 0 && finalX < width && finalY >= 0 && finalY < height) {
// 			atomicAdd(sum + angle, (int) WEIGHTS[angle] * image[finalY * width + finalX]);
// 		}
// 	}
// }

__global__ void getAngleChange(Particle * particles, uint32_t particleCount, uint32_t width, uint32_t height, uint32_t * image, uint64_t * rand) {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < particleCount) {
		float deltaAngle[DELTAANGLES] = {0, -SCANANGLE, SCANANGLE};
		float turnAngles[DELTAANGLES] = {0, -MAXTURN, MAXTURN};
		uint32_t offX;
		uint32_t offY;
		int32_t tempX;
		int32_t tempY;
		uint32_t sum;
		// int sum[DELTAANGLES] = {0};
		uint64_t maxSum = 0;
		float bestAngle = 0;

		// uint32_t fidelity = DELTAANGLES * 2 * SCANRADIUS;
		// uint32_t blocks = (fidelity + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
		// parallelParticleScan <<< blocks, THREADSPERBLOCK >>> (index, fidelity, 2 * SCANRADIUS, sum, particles, width, height, image);
		// cudaDeviceSynchronize();
		// for(int i = 0; i < DELTAANGLES; i++) {
		// 	bestAngle = sum[i] > maxSum ? turnAngles[i] : bestAngle;
		// 	maxSum = sum[i] > maxSum ? sum[i] : maxSum;
		// }

		for(int i = 0; i < 3; i++) {
			sum = 0;
			offX = (uint32_t) (particles[index].posX + SCANOFFSET * cos(particles[index].angle + deltaAngle[i]));
			offY = (uint32_t) (particles[index].posY + SCANOFFSET * sin(particles[index].angle + deltaAngle[i]));
			for(int j = -SCANRADIUS; j < SCANRADIUS; j++) {
				tempY = offY + j;
				for(int k = -SCANRADIUS; k < SCANRADIUS; k++) {
					tempX = offX + k;
					if(tempX >= 0 && tempX < width && tempY >= 0 && tempY < height) {
						sum += WEIGHTS[i] * image[tempY * width + tempX];
					}
				}
			}
			bestAngle = sum > maxSum ? turnAngles[i] : bestAngle;
			maxSum = sum > maxSum ? sum : maxSum;
		}
		particles[index].angle += bestAngle;// * (genRand(&rand[index]) + 1)/2;
	}
}

__global__ void fadeColor(uint32_t * image, uint32_t width, uint32_t height) {
	uint32_t index = blockIdx.x *blockDim.x + threadIdx.x;

	if(index < (width * height)) {
		if(image[index] > COLORFADERATE) {
			image[index] -= COLORFADERATE;
		} else {
			image[index] = 0;
		}
	}
}

__global__ void blur(uint32_t * image, uint32_t width, uint32_t height) {
	uint32_t index = blockIdx.x *blockDim.x + threadIdx.x;

	if(index < (width * height)) {
		int32_t tempX;
		int32_t tempY;
		uint32_t averageColor[3] = {0, 0, 0};
		for(int x = -1; x <= 1; x++) {
			tempX = (index % width) + x;
			for(int y = -1; y <= 1; y++) {
				tempY = (index / width) + y;
				if(tempX >= 0 && tempX < width && tempY >= 0 && tempY < height) {
					for(int i = 0; i < 3; i++) {
						averageColor[i] += image[tempY * width + tempX] >> (i * 8) & 0xff;
					}
				}
			}
		}

		image[index] = 0;
		for(int i = 0; i < 3; i++) {
			image[index] |= (averageColor[i]/9) << (8 * i);
		}
	}
}

Animation::Animation(uint32_t widthIn, uint32_t heightIn, uint32_t particlesIn) {
	width = widthIn;
	height = heightIn;
	particleCount = particlesIn;
	// TODO: test if using passed in variables is better than class variables
	particles = (Particle *) gpuMemAlloc(particleCount * sizeof(Particle));
	imageSize = width * height * sizeof(uint32_t);
	d_image = (uint32_t *) gpuMemAlloc(imageSize);
	image = (uint32_t *) malloc(imageSize);
	blockCountGPU = (particleCount + THREADSPERBLOCK - 1)/THREADSPERBLOCK;
	init();
}

void Animation::init() {
	h_rand = (uint64_t *) malloc(particleCount * sizeof(uint64_t));
	d_rand = (uint64_t *) gpuMemAlloc(particleCount * sizeof(uint64_t));
	srand(0);
	for(int i = 0; i < particleCount; i++) {
		h_rand[i] = (((uint64_t) rand()) << 48) | (((uint64_t) rand()) << 32) | (((uint64_t) rand()) << 16) | (uint64_t) rand();
	}
	cudaMemcpy(d_rand, h_rand, particleCount * sizeof(uint64_t), cudaMemcpyHostToDevice);
	randomizeValues <<< blockCountGPU, THREADSPERBLOCK >>> (particles, particleCount, width, height, d_rand);
}

void Animation::nextFrame() {
	// cudaMemset(d_image, 0x00000000, imageSize);
	nextFrameGPU <<< blockCountGPU, THREADSPERBLOCK >>> (particles, particleCount, width, height, d_image, d_rand);
	blur <<< (width * height + THREADSPERBLOCK - 1)/THREADSPERBLOCK, THREADSPERBLOCK >>> (d_image, width, height);
	getAngleChange <<< blockCountGPU, THREADSPERBLOCK >>> (particles, particleCount, width, height, d_image, d_rand);
	fadeColor <<< (width * height + THREADSPERBLOCK - 1)/THREADSPERBLOCK, THREADSPERBLOCK >>> (d_image, width, height);
}

void Animation::exit() {
	cudaFree(d_image);
	cudaFree(particles);
}

void * Animation::getImage() {
	cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost);
	return image;
}