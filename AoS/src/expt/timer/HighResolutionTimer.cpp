#include "HighResolutionTimer.h"


HighResolutionTimer::HighResolutionTimer() = default;

HighResolutionTimer::~HighResolutionTimer() = default;

std::chrono::duration<double> HighResolutionTimer::now() {
    return std::chrono::high_resolution_clock::now().time_since_epoch();
}