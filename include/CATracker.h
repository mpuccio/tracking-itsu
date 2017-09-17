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
  public:
    void computeLayerTracklets(CAPrimaryVertexContext&);
    void computeLayerCells(CAPrimaryVertexContext&);

  protected:
    ~CATrackerTraits() = default;
};

template<bool IsGPU>
class CATracker: private CATrackerTraits<IsGPU>
{
  private:
    typedef CATrackerTraits<IsGPU> TrackerTraits;

  public:
    CATracker();

    CATracker(const CATracker&) = delete;
    CATracker &operator=(const CATracker&) = delete;

    std::vector<std::vector<CARoad>> clustersToTracks(const CAEvent&);
    std::vector<std::vector<CARoad>> clustersToTracksVerbose(const CAEvent&);
    std::vector<std::vector<CARoad>> clustersToTracksMemoryBenchmark(const CAEvent&, std::ofstream&);
    std::vector<std::vector<CARoad>> clustersToTracksTimeBenchmark(const CAEvent&, std::ofstream&);

  protected:
    void computeTracklets();
    void computeCells();
    void findCellsNeighbours();
    void findTracks();
    void traverseCellsTree(const int, const int);
    void computeMontecarloLabels();

  private:
    void evaluateTask(void (CATracker<IsGPU>::*)(void), const char*);
    void evaluateTask(void (CATracker<IsGPU>::*)(void), const char*, std::ostream&);

    CAPrimaryVertexContext mPrimaryVertexContext;
};

#endif /* TRACKINGITSU_INCLUDE_CATRACKER_H_ */
