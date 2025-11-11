#include "SteadyTimer.h"


SteadyTimer::SteadyTimer() = default;

SteadyTimer::~SteadyTimer() = default;

std::chrono::duration<double> SteadyTimer::now() {
    return std::chrono::steady_clock::now().time_since_epoch();
}