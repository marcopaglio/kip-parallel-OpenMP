#ifndef STEADYTIMER_H
#define STEADYTIMER_H
#include "Timer.h"


/**
 * A final concrete implementation of the @ref Timer interface for time measurement
 * using steady timing functionalities based on `std::chrono::steady_clock`.
 */
class SteadyTimer final : public Timer {
public:
    /**
     * Default constructor.
     */
    SteadyTimer();

    /**
     * Default destructor.
     */
    ~SteadyTimer() override;

    /**
     * Retrieves the current time as a duration since the steady clock's epoch.
     *
     * @return The current time, represented as a `std::chrono::duration<double>`.
     */
    std::chrono::duration<double> now() override;
};


#endif //STEADYTIMER_H
