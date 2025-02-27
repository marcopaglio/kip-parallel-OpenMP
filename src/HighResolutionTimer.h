#ifndef HIGHRESOLUTIONTIMER_H
#define HIGHRESOLUTIONTIMER_H
#include "Timer.h"


class HighResolutionTimer final : public Timer {
public:
    HighResolutionTimer();
    ~HighResolutionTimer() override;

    std::chrono::duration<double> now() override;
};



#endif //HIGHRESOLUTIONTIMER_H
