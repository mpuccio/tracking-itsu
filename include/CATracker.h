/// \file CATracker.h
/// \brief 
///
/// \author Iacopo Colonnelli, Politecnico di Torino
///
/// \copyright Copyright (C) 2017  Iacopo Colonnelli. \n\n
///   This program is free software: you can redistribute it and/or modify
///   it under the terms of the GNU General Public License as published by
///   the Free Software Foundation, either version 3 of the License, or
///   (at your option) any later version. \n\n
///   This program is distributed in the hope that it will be useful,
///   but WITHOUT ANY WARRANTY; without even the implied warranty of
///   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
///   GNU General Public License for more details. \n\n
///   You should have received a copy of the GNU General Public License
///   along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////

#ifndef TRACKINGITSU_INCLUDE_CATRACKER_H_
#define TRACKINGITSU_INCLUDE_CATRACKER_H_

#include <array>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "CADefinitions.h"
#include "CAEvent.h"
#include "CAMathUtils.h"
#include "CAPrimaryVertexContext.h"
#include "CARoad.h"

template<bool IsGPU>
class CATrackerTraits
{
  private:
    typedef CAPrimaryVertexContext<IsGPU> Context;
  public:
    void computeLayerTracklets(Context&, const int);
    void postProcessTracklets(Context&);
    void computeLayerCells(Context&, const int);
    void postProcessCells(Context&);

  protected:
    ~CATrackerTraits() = default;
};

template<bool IsGPU>
class CATracker: private CATrackerTraits<IsGPU>
{
  private:
    typedef CATrackerTraits<IsGPU> TrackerTraits;
    typedef CAPrimaryVertexContext<IsGPU> TrackerContext;

  public:
    explicit CATracker(const CAEvent&);

    CATracker(const CATracker&) = delete;
    CATracker &operator=(const CATracker&) = delete;

    std::vector<std::vector<CARoad>> clustersToTracks();
    std::vector<std::vector<CARoad>> clustersToTracksVerbose();
    std::vector<std::vector<CARoad>> clustersToTracksMemoryBenchmark(std::ofstream&);
    std::vector<std::vector<CARoad>> clustersToTracksTimeBenchmark(std::ofstream&);

  protected:
    void computeTracklets(TrackerContext&);
    void computeCells(TrackerContext&);
    void findCellsNeighbours(TrackerContext&);
    void findTracks(TrackerContext&);
    void traverseCellsTree(TrackerContext&, const int, const int);
    void computeMontecarloLabels(TrackerContext&);

  private:
    void evaluateTask(void (CATracker<IsGPU>::*)(TrackerContext&), const char*, TrackerContext&);
    void evaluateTask(void (CATracker<IsGPU>::*)(TrackerContext&), const char*, TrackerContext&, std::ostream&);

    const CAEvent& mEvent;
    std::vector<int> mUsedClustersTable;
};

#endif /* TRACKINGITSU_INCLUDE_CATRACKER_H_ */
