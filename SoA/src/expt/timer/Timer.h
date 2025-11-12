#ifndef TIMER_H
#define TIMER_H
#include <chrono>


/**
 * Represents an interface for time measurement.
 */
class Timer {
public:
    /**
     * Default constructor.
     */
    Timer();

    /**
     * Default destructor.
     *
     * It is declared as virtual to ensure that the proper destructor is called for derived classes.
     */
    virtual ~Timer();

    /**
     * Retrieves the current time as a duration.
     *
     * This method is pure virtual and must be implemented by derived classes.
     *
     * @return The current time represented as a `std::chrono::duration<double>`.
     */
    virtual std::chrono::duration<double> now() = 0;
};


#endif //TIMER_H
