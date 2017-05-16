/// \file CAIndexTable.h
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

#ifndef TRACKINGITSU_INCLUDE_CALOOKUPTABLE_H_
#define TRACKINGITSU_INCLUDE_CALOOKUPTABLE_H_

#include <utility>
#include <vector>

#include "CACluster.h"
#include "CAConstants.h"
#include "CADefinitions.h"

#if defined(TRACKINGITSU_GPU_MODE)
# include "CAGPUArray.h"
#endif

namespace TRACKINGITSU_TARGET_NAMESPACE {
class CAIndexTable
  final
  {
    public:
      CAIndexTable();
      CAIndexTable(const int, const std::vector<CACluster>&);

      GPU_DEVICE int getBinFirstClusterIndex(const int) const;
      const std::array<int, 4> getSelectedBinsRect(const float, const float, const float, const float) const;
      const std::vector<std::pair<int,int>> selectClusters(const std::array<int, 4>&) const;

    private:
      int mLayerIndex;
      GPU_ARRAY<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1> mTableBins;
  };

  GPU_DEVICE inline int CAIndexTable::getBinFirstClusterIndex(const int binIndex) const
  {
	  return mTableBins[binIndex];
  }
}

#endif /* TRACKINGITSU_INCLUDE_CALOOKUPTABLE_H_ */
