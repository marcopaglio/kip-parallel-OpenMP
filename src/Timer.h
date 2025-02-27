#ifndef TIMER_H
#define TIMER_H
#include <chrono>


class Timer {
public:
    Timer();
    virtual ~Timer();

    virtual std::chrono::duration<double> now() = 0;
};


#endif //TIMER_H
