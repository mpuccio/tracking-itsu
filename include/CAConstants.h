/// \file CAConstants.h
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

#ifndef TRACKINGITSU_INCLUDE_CACONSTANTS_H_
#define TRACKINGITSU_INCLUDE_CACONSTANTS_H_

#include <array>

namespace CAConstants {

namespace Math {
constexpr float Pi { 3.14159265359 };
constexpr float TwoPi { 2.0 * Pi };
constexpr float FloatMinThreshold { 1e-20f };
}

namespace ITS {
constexpr int LayersNumber { 7 };
constexpr int TrackletsPerRoad { 6 };
constexpr int CellsPerRoad { LayersNumber - 2 };
constexpr int TracksReconstructionIterations { 2 };
constexpr int UnusedIndex { -1 };
}

namespace Thresholds {
constexpr std::array<float, ITS::LayersNumber> LayersRCoordinate { { 2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f,
    34.388f, 39.3329f } };
constexpr std::array<std::array<float, ITS::TrackletsPerRoad>, ITS::TracksReconstructionIterations> TrackletMaxDeltaZThreshold {
    { { 0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f }, { 1.0f, 1.0f, 1.5f, 1.5f, 1.5f, 1.5f } } };
constexpr std::array<float, ITS::TracksReconstructionIterations> CellMaxDeltaTanLambdaThreshold { { 0.025f, 0.05f } };
constexpr std::array<std::array<float, ITS::CellsPerRoad>, ITS::TracksReconstructionIterations> CellMaxDeltaZThreshold {
    { { 0.2f, 0.4f, 0.5f, 0.6f, 3.0f }, { 1.0f, 0.4f, 0.4f, 1.5f, 3.0f } } };
constexpr std::array<std::array<float, ITS::CellsPerRoad>, ITS::TracksReconstructionIterations> CellMaxDistanceOfClosestApproachThreshold {
    { { 0.05f, 0.04f, 0.05f, 0.2f, 0.4f }, { 1.0f, 0.4f, 0.4f, 1.5f, 3.0f } } };
constexpr std::array<float, ITS::TracksReconstructionIterations> CellMaxDeltaPhiThreshold { { 0.14f, 0.2f } };
constexpr float ZCoordinateCut { 0.5f };
constexpr std::array<float, ITS::TracksReconstructionIterations> PhiCoordinateCut { 1.0f, 3.0f };
constexpr std::array<std::array<float, ITS::CellsPerRoad - 1>, ITS::TracksReconstructionIterations> NeighbourCellMaxNormalVectorsDelta {
    { { 0.002f, 0.009f, 0.002f, 0.005f }, { 0.005f, 0.0035f, 0.009f, 0.03f } } };
constexpr std::array<std::array<float, ITS::CellsPerRoad - 1>, ITS::TracksReconstructionIterations> NeighbourCellMaxCurvaturesDelta {
    { { 0.008f, 0.0025f, 0.003f, 0.0035f }, { 0.02f, 0.005f, 0.006f, 0.007f } } };
constexpr std::array<int, ITS::TracksReconstructionIterations> TracksMinLength { 4, 2 };
}

namespace IndexTable {
constexpr int ZBins { 20 };
constexpr int PhiBins { 20 };
}

}

#endif /* TRACKINGITSU_INCLUDE_CACONSTANTS_H_ */
