#ifndef HIGHRESOLUTIONTIMER_H
#define HIGHRESOLUTIONTIMER_H
#include "Timer.h"


/**
 * A final concrete implementation of the @ref Timer interface for time measurement
 * using high-resolution timing functionalities based on `std::chrono::high_resolution_clock`.
 */
class HighResolutionTimer final : public Timer {
public:
    /**
     * Default constructor.
     */
    HighResolutionTimer();

    /**
     * Default destructor.
     */
    ~HighResolutionTimer() override;

    /**
     * Retrieves the current high-resolution time as a duration.
     *
     * @return The current high-resolution time as a `std::chrono::duration<double>`.
     */
    std::chrono::duration<double> now() override;
};



#endif //HIGHRESOLUTIONTIMER_H
