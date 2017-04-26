/// \file CAArrayUtils.h
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

#ifndef TRACKINGITSU_INCLUDE_CAARRAYUTILS_H_
#define TRACKINGITSU_INCLUDE_CAARRAYUTILS_H_

#include <array>
#include <utility>

namespace CAArrayUtils {
template<typename T, std::size_t ...Is, typename Initializer>
constexpr std::array<T, sizeof...(Is)> fillArray(Initializer, std::index_sequence<Is...>);
template<typename T, std::size_t N, typename Initializer>
constexpr std::array<T, N> fillArray(Initializer);
}

template<typename T, std::size_t ...Is, typename Initializer>
constexpr std::array<T, sizeof...(Is)> CAArrayUtils::fillArray(Initializer initializer, std::index_sequence<Is...>)
{
  return std::array<T, sizeof...(Is)> { { initializer(Is)... } };
}

template<typename T, std::size_t N, typename Initializer>
constexpr std::array<T, N> CAArrayUtils::fillArray(Initializer initializer)
{
  return CAArrayUtils::fillArray<T>(initializer, std::make_index_sequence<N> { });
}

#endif /* TRACKINGITSU_INCLUDE_CAARRAYUTILS_H_ */
