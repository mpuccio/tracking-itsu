/// \file Tracker.h
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

#ifndef TRACKINGITSU_INCLUDE_TRACKER_H_
#define TRACKINGITSU_INCLUDE_TRACKER_H_

#include <array>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/MathUtils.h"
#include "ITSReconstruction/CA/PrimaryVertexContext.h"
#include "ITSReconstruction/CA/Road.h"

namespace o2
{
namespace ITS
{
namespace CA
{

template<bool IsGPU>
class TrackerTraits
{
  public:
    void computeLayerTracklets(PrimaryVertexContext&);
    void computeLayerCells(PrimaryVertexContext&);

  protected:
    ~TrackerTraits() = default;
};

template<bool IsGPU>
class Tracker: private TrackerTraits<IsGPU>
{
  private:
    typedef TrackerTraits<IsGPU> Trait;

  public:
    Tracker();

    Tracker(const Tracker&) = delete;
    Tracker &operator=(const Tracker&) = delete;

    std::vector<std::vector<Road>> clustersToTracks(const Event&);
    std::vector<std::vector<Road>> clustersToTracksVerbose(const Event&);
    std::vector<std::vector<Road>> clustersToTracksMemoryBenchmark(const Event&, std::ofstream&);
    std::vector<std::vector<Road>> clustersToTracksTimeBenchmark(const Event&, std::ofstream&);

  protected:
    void computeTracklets();
    void computeCells();
    void findCellsNeighbours();
    void findTracks();
    void traverseCellsTree(const int, const int);
    void computeMontecarloLabels();

  private:
    void evaluateTask(void (Tracker<IsGPU>::*)(void), const char*);
    void evaluateTask(void (Tracker<IsGPU>::*)(void), const char*, std::ostream&);

    PrimaryVertexContext mPrimaryVertexContext;
};

}
}
}

#endif /* TRACKINGITSU_INCLUDE_TRACKER_H_ */
