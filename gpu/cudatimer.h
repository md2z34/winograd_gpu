#pragma once
#include <cuda_runtime_api.h>

class cudaTimer
{
private:

	float elapsed_time;

	cudaStream_t streamID;
	cudaEvent_t  start_event, stop_event;

	bool isStarted;

public:

	cudaTimer() : isStarted(false) {};

	~cudaTimer() { if (isStarted) { cudaEventDestroy(start_event); cudaEventDestroy(stop_event); } }

	void start(cudaStream_t streamNo = 0)
	{
		if (!isStarted)
		{
			streamID = streamNo;

			cudaEventCreate(&start_event); cudaEventCreate(&stop_event);
			cudaEventRecord(start_event, streamID);

			isStarted = true;
		}
		else
			;//printf(" ERROR: Timer already initialized!\n");
	}

	void stop()
	{
		if (isStarted)
		{
			cudaEventRecord(stop_event, streamID);
			cudaEventSynchronize(stop_event);
			cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

			cudaEventDestroy(start_event); cudaEventDestroy(stop_event);

			isStarted = false;
		}
		else
			;//printf(" ERROR: Timer not even started!\n");
	}

	float elapsedTime() { if (!isStarted) return elapsed_time; else return -1.f; }

	bool  isRunning() { return isStarted; }
};
