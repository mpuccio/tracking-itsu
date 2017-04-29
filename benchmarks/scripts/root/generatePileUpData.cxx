/// \file generatePileUpData.cxx
/// \brief
///
/// \author Iacopo Colonnelli, Politecnico di Torino

/***************************************************************************
 *  Copyright (C) 2017  Iacopo Colonnelli
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ***************************************************************************/

#include <fstream>
#include <string>

namespace {
constexpr int PrimaryVertexLayerId{ -1  };
constexpr int EventsFileColumnsNum{ 10 };
constexpr int EventsFileMonteCarloIndex{ 8 };
constexpr int LabelsFileColumnsNum{ 6 };
}

int roundUp(const int numToRound, const int multiple);
void mergeEvents(const int pileUp, const std::string& inputFolder, const std::string& outputFolder);
void mergeLabels(const int pileUp, const std::string& inputFolder, const std::string& outputFolder);
void generatePileUpData(const int pileUp, const std::string& inputFolder);
void generatePileUpData(const int pileUp, const std::string& inputFolder, const std::string& outputFolder);

int roundUp(const int numToRound, const int multiple) {

  if(multiple == 0) {

    return numToRound;
  }

  int remainder = numToRound % multiple;

  if(remainder == 0) {

    return numToRound;
  }

  return numToRound + multiple - remainder;
}

void mergeEvents(const int pileUp, const std::string& inputFolder, const std::string& outputFolder) {

  std::ifstream inputStream;
  std::ofstream outputStream;
  std::string line;
  int layerId, monteCarloId, verticesNum = 0, maxMonteCarloLabel = 0, monteCarloOffset = 0;
  std::vector<std::string> clusterLines;
  std::array<std::string, EventsFileColumnsNum> lineData;

  inputStream.open(inputFolder + "data.txt");
  outputStream.open(outputFolder + "merged_data.txt");


  while (std::getline(inputStream, line)) {

    std::istringstream inputStringStream(line);

    if (inputStringStream >> layerId) {

      if (layerId == PrimaryVertexLayerId) {

        maxMonteCarloLabel = roundUp(maxMonteCarloLabel, 100000);
        monteCarloOffset += maxMonteCarloLabel;
        maxMonteCarloLabel = 0;

        if(verticesNum == pileUp) {

          const int linesNum = clusterLines.size();
          for(int iLine = 0; iLine < linesNum; ++iLine) {

            outputStream << clusterLines[iLine];
          }

          verticesNum = 0;
          monteCarloOffset = 0;
          clusterLines.clear();
        }

        outputStream << line;
        ++verticesNum;

      } else {

        lineData[0] = layerId;

        for(int iValue = 1; iValue < EventsFileColumnsNum; ++iValue) {

          inputStringStream >> lineData[iValue];
        }

        monteCarloId = std::stoi(lineData[EventsFileMonteCarloIndex]);

        if(monteCarloId > maxMonteCarloLabel) {

          maxMonteCarloLabel = monteCarloId;
        }

        lineData[EventsFileMonteCarloIndex] = monteCarloId + monteCarloOffset;

        clusterLines.push_back(line);
      }
    }
  }

  std::cout << "Generated " << outputFolder << "merged_data.txt with pile-up " << pileUp << std::endl;
}

void mergeLabels(const int pileUp, const std::string& inputFolder, const std::string& outputFolder) {

  std::ifstream inputStream;
  std::ofstream outputStream;
  std::string line;
  int monteCarloId, verticesNum = 0, maxMonteCarloLabel = 0, monteCarloOffset = 0;
  std::vector<std::string> clusterLines;
  std::array<std::string, LabelsFileColumnsNum> lineData;

  inputStream.open(inputFolder + "labels.txt");
  outputStream.open(outputFolder + "merged_labels.txt");


  while (std::getline(inputStream, line)) {

    std::istringstream inputStringStream(line);

    if (inputStringStream >> monteCarloId) {

      if (monteCarloId == PrimaryVertexLayerId) {

        maxMonteCarloLabel = roundUp(maxMonteCarloLabel, 100000);
        monteCarloOffset += maxMonteCarloLabel;
        maxMonteCarloLabel = 0;

        if(verticesNum == pileUp) {

          outputStream << line;

          const int linesNum = clusterLines.size();
          for(int iLine = 0; iLine < linesNum; ++iLine) {

            outputStream << clusterLines[iLine];
          }

          verticesNum = 0;
          monteCarloOffset = 0;
          clusterLines.clear();
        }

        ++verticesNum;

      } else {

        lineData[0] = monteCarloId;

        for(int iValue = 1; iValue < LabelsFileColumnsNum; ++iValue) {

          inputStringStream >> lineData[iValue];
        }

        if(monteCarloId > maxMonteCarloLabel) {

          maxMonteCarloLabel = monteCarloId;
        }

        lineData[0] = monteCarloId + monteCarloOffset;

        clusterLines.push_back(line);
      }
    }
  }

  std::cout << "Generated " << outputFolder << "merged_data.txt with pile-up " << pileUp << std::endl;
}

void generatePileUpData(const int pileUp, const std::string& inputFolder) {

  generatePileUpData(pileUp, inputFolder, inputFolder);
}

void generatePileUpData(const int pileUp, const std::string& inputFolder, const std::string& outputFolder) {

  mergeEvents(pileUp, inputFolder, outputFolder);
  mergeLabels(pileUp, inputFolder, outputFolder);
}

int main(int argc, char** argv) {

  int pileUp;
  std::string inputFolder { "" };
  std::string outputFolder { "" };

  if (argv[1] != NULL) {

    pileUp = std::stoi(argv[1]);

  } else {

    std::cout << "generatePileUpData.cxx pileUp inputFolder [outputFolder=inputFolder]" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (argv[2] != NULL) {

    inputFolder = std::string(argv[2]);
  }

  if (argv[3] != NULL) {

    outputFolder = std::string(argv[3]);

  } else if (argv[2] != NULL) {

    outputFolder = inputFolder;
  }

  generatePileUpData(pileUp, inputFolder, outputFolder);
  return 0;
}
