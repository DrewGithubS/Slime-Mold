#include <time.h>
#include <unistd.h>

#include "Animation.h"
#include "Renderer.h"

const uint32_t WIDTH = 1920;
const uint32_t HEIGHT = 977;
const uint32_t PARTICLES = 100000;
const uint32_t FRAMESPERSECOND = 144;
const uint32_t MICROSECONDSPERSECOND = 1000 * 1000;
const uint32_t NANOSECONDSPERMICROSECOND = 1000;

uint64_t getMicrosecondsPassed(struct timespec start, struct timespec end) {
	return ((end.tv_sec - start.tv_sec) * MICROSECONDSPERSECOND) + ((end.tv_nsec - start.tv_nsec) / NANOSECONDSPERMICROSECOND);
}


int main() {
	Renderer * renderer = new Renderer(WIDTH, HEIGHT);
	Animation * animation = new Animation(WIDTH, HEIGHT, PARTICLES);

	uint32_t microsecondsPerFrame = MICROSECONDSPERSECOND / FRAMESPERSECOND;
	struct timespec start;
	struct timespec end;

	clock_gettime(CLOCK_REALTIME, &start);
	usleep(microsecondsPerFrame);
	uint32_t microsecondsPassed;

	int frame[WIDTH * HEIGHT];

	for(int i = 0; i < WIDTH * HEIGHT; i++) {
		// frame[i] = i % 2 ? 0x00aaaaff : 0x00000000;
		frame[i] = 0x0022ffff;
	}
	while(1) {
		clock_gettime(CLOCK_REALTIME, &end);
		microsecondsPassed = getMicrosecondsPassed(start, end);
		if(microsecondsPassed >= microsecondsPerFrame) {
			renderer->checkForEvent();
			// renderer->getFrame();
			animation->nextFrame();
			renderer->setFrame(animation->getImage());
			renderer->render();
		} else {
			usleep(microsecondsPerFrame - microsecondsPassed);
		}
		// break;
	}

}