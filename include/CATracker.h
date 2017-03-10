/// \file CATracker.h
/// \brief 
///
/// \author Iacopo Colonnelli, Politecnico di Torino

#ifndef TRACKINGITSU_INCLUDE_CATRACKER_H_
#define TRACKINGITSU_INCLUDE_CATRACKER_H_

class CATracker final
{
  public:
    CATracker();

    CATracker(const CATracker&) = delete;
    CATracker &operator=(const CATracker &tr) = delete;

//    int ClustersToTracks(AliESDEvent *event);
//    int LoadClusters(TTree *ct);
//    void UnloadClusters();
//    AliCluster *GetCluster(int index) const;
};

#endif /* TRACKINGITSU_INCLUDE_CATRACKER_H_ */
