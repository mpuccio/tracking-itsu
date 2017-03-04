/// \file CATracker.h
/// \brief 
///
/// \author Iacopo Colonnelli, Politecnico di Torino

#ifndef TRACKINGITSU_INCLUDE_CATRACKER_H_
#define TRACKINGITSU_INCLUDE_CATRACKER_H_

class CATracker
{
  public:
    CATracker();
    virtual ~CATracker();

    CATracker(const CATracker&) = delete;
    CATracker &operator=(const CATracker &tr) = delete;
};

#endif /* TRACKINGITSU_INCLUDE_CATRACKER_H_ */
