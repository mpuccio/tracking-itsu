/// \file IndexTableUtils.cxx
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

#include "ITSReconstruction/CA/IndexTableUtils.h"

namespace o2
{
namespace ITS
{
namespace CA
{

const std::vector<std::pair<int, int>> IndexTableUtils::selectClusters(
    const std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1> &indexTable,
    const std::array<int, 4> &selectedBinsRect)
{
  std::vector<std::pair<int, int>> filteredBins { };

  int phiBinsNum { selectedBinsRect[3] - selectedBinsRect[1] + 1 };

  if (phiBinsNum < 0) {

    phiBinsNum += Constants::IndexTable::PhiBins;
  }

  filteredBins.reserve(phiBinsNum);

  for (int iPhiBin { selectedBinsRect[1] }, iPhiCount { 0 }; iPhiCount < phiBinsNum;
      iPhiBin = ++iPhiBin == Constants::IndexTable::PhiBins ? 0 : iPhiBin, iPhiCount++) {

    const int firstBinIndex { IndexTableUtils::getBinIndex(selectedBinsRect[0], iPhiBin) };

    filteredBins.emplace_back(indexTable[firstBinIndex],
        countRowSelectedBins(indexTable, iPhiBin, selectedBinsRect[0], selectedBinsRect[2]));
  }

  return filteredBins;
}

}
}
}
