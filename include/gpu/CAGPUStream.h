/// \file CAGPUStream.h
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

#ifndef TRAKINGITSU_INCLUDE_GPU_STREAM_H_
#define TRAKINGITSU_INCLUDE_GPU_STREAM_H_

#include "CADefinitions.h"

class CAGPUStream
  final {

    public:
      CAGPUStream();
      ~CAGPUStream();

      CAGPUStream(const CAGPUStream&) = delete;
      CAGPUStream &operator=(const CAGPUStream&) = delete;

      const GPUStream& get() const;

    private:
      GPUStream mStream;
  };

#endif /* TRAKINGITSU_INCLUDE_GPU_STREAM_H_ */
