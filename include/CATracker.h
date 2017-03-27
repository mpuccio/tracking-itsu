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

#include <vector>

#include "CACell.h"
#include "CAConstants.h"
#include "CAEvent.h"
#include "CAIndexTable.h"
#include "CATracklet.h"

class CATracker final
{
  public:
    explicit CATracker(const CAEvent&);

    CATracker(const CATracker&) = delete;
    CATracker &operator=(const CATracker&) = delete;

    int clustersToTracks();

  protected:
    void computeTracklets(const int, std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad>&,
        std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad>&);
    void computeCells(const int);
    void findCellsNeighbours(const int, std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad>&,
        const std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1>&);

  private:
    const CAEvent& mEvent;
    std::vector<bool> mUsedClustersTable;
    std::array<CAIndexTable, CAConstants::ITS::LayersNumber> mIndexTables;
};

#endif /* TRACKINGITSU_INCLUDE_CATRACKER_H_ */
