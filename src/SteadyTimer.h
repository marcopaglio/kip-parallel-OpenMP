#ifndef STEADYTIMER_H
#define STEADYTIMER_H
#include "Timer.h"


class SteadyTimer final : public Timer {
public:
    SteadyTimer();
    ~SteadyTimer() override;

    std::chrono::duration<double> now() override;
};


#endif //STEADYTIMER_H
