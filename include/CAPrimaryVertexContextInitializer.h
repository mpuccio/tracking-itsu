/// \file CAPrimaryVertexContextInitializer.h
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
#ifndef TRACKINGITSU_INCLUDE_CAPRIMARYVERTEXCONTEXTINITIALIZER_H_
#define TRACKINGITSU_INCLUDE_CAPRIMARYVERTEXCONTEXTINITIALIZER_H_

#include <array>
#include <vector>

#include "CACell.h"
#include "CACluster.h"
#include "CAConstants.h"
#include "CAEvent.h"
#include "CATracklet.h"

struct CAPrimaryVertexContextInitializer
    final
    {
      static std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber> initClusters(const CAEvent&, const int);
      static std::array<std::array<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
          CAConstants::ITS::TrackletsPerRoad> initIndexTables(
          std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber>&);
      static std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad> initTracklets(const CAEvent&);
      static std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad> initTrackletsLookupTable(const CAEvent&);
      static std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad> initCells(const CAEvent&);
      static std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1> initCellsLookupTable(const CAEvent&);
  };

#endif //TRACKINGITSU_INCLUDE_CAPRIMARYVERTEXCONTEXTINITIALIZER_H_
