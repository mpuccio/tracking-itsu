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

namespace MathConstants {
constexpr float Pi = 3.14159265359;
constexpr float TwoPi = 2.0 * Pi;
}

namespace ITSConstants {
constexpr int LayersNumber = 7;
constexpr int TrackletsPerRoad = 6;
constexpr int CellsPerRoad = LayersNumber - 2;
constexpr int LookupTableZBins = 20;
constexpr int LookupTablePhiBins = 20;
}

#endif /* TRACKINGITSU_INCLUDE_CACONSTANTS_H_ */
