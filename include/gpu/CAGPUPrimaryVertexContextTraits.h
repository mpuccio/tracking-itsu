/// \file CAGPUPrimaryVertexContextTraits.h
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
#ifndef TRACKINGITSU_INCLUDE_GPU_CATRACKERCONTEXTTRAITS_H_
#define TRACKINGITSU_INCLUDE_GPU_CATRACKERCONTEXTTRAITS_H_

#include <array>

#include "CAGPUVector.h"
#include "CAPrimaryVertexContext.h"
#include "CATracklet.h"

using namespace TRACKINGITSU_TARGET_NAMESPACE;

struct CAGPUPrimaryVertexContextTraits {
    explicit CAGPUPrimaryVertexContextTraits(const CAPrimaryVertexContext<TRACKINGITSU_GPU_MODE>&);
    ~CAGPUPrimaryVertexContextTraits();

    CAGPUPrimaryVertexContextTraits(const CAGPUPrimaryVertexContextTraits&) = delete;
    CAGPUPrimaryVertexContextTraits &operator=(const CAGPUPrimaryVertexContextTraits&) = delete;

    std::array<CAGPUVector<CACluster>, CAConstants::ITS::LayersNumber> dClusters;
    std::array<CAGPUVector<int>, CAConstants::ITS::TrackletsPerRoad> dIndexTables;
    std::array<CAGPUVector<CATracklet>, CAConstants::ITS::TrackletsPerRoad> dTracklets;
    std::array<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad> dTrackletsLookupTable;
};

#endif /* TRACKINGITSU_INCLUDE_GPU_CATRACKERCONTEXTTRAITS_H_ */
