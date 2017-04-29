/// \file plotMemoryOccupancyBenchmark.cxx
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

#include <array>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TGraph.h"
#include "TROOT.h"

namespace{
constexpr int BinNumber{ 1000 };
constexpr int LayersNumber{ 7 };
constexpr int TrackletsNumber{ 6 };
constexpr int CellsNumber{ 5 };
constexpr int RoadsNumber{ 1 };
constexpr int DataTypes{ 4 };
constexpr int ClustersDataType{ 0 };
constexpr int TrackletsDataType{ 1 };
constexpr int CellsDataType{ 2 };
constexpr int RoadsDataType{ 3 };
constexpr std::array<int, DataTypes> LineDataNum{ LayersNumber, TrackletsNumber, CellsNumber, RoadsNumber };
}

std::vector<std::array<std::vector<int>, DataTypes>> loadData(const std::string& fileName)
{
  std::ifstream inputStream;
  std::string line;
  std::vector<std::array<std::vector<int>, DataTypes>> dataReport;
  std::array<std::vector<int>, DataTypes> eventReport;
  int lineType = 0, currentDataValue;

  inputStream.open(fileName);

  while (std::getline(inputStream, line)) {

    std::istringstream inputStringStream(line);
    std::vector<int> currentLineData;
    const int currentLineDataNum = LineDataNum[lineType];
    currentLineData.reserve(currentLineDataNum);

    for(int iValue = 0; iValue < currentLineDataNum; ++iValue) {

      inputStringStream >> currentDataValue;
      currentLineData.emplace_back(currentDataValue);
    }

    eventReport[lineType] = currentLineData;
    ++lineType;

    if(lineType == DataTypes) {

      lineType = 0;
      dataReport.emplace_back(eventReport);
    }
  }

  return dataReport;
}

void plotHistogram(TH1F& histogram, std::vector<std::array<std::vector<int>, DataTypes>>& dataReport,
    const int dataType, const int layerIndex, const std::string& outputFileName) {

  TCanvas graphCanvas { };
  graphCanvas.SetGrid();

  const int numEvents = dataReport.size();

  for(int iEvent = 0; iEvent < numEvents; ++iEvent) {

    const int currentValue = dataReport[iEvent][dataType][layerIndex];
    const int complexity = LayersNumber + 1 - LineDataNum[dataType];
    long clustersProduct = 1;

    for(int iClustersLayer = layerIndex; iClustersLayer < layerIndex + complexity; ++iClustersLayer) {

      clustersProduct *= dataReport[iEvent][ClustersDataType][iClustersLayer];
    }

    histogram.Fill(static_cast<double>(currentValue) / clustersProduct);
  }

  histogram.Draw();

  graphCanvas.Print(outputFileName.c_str());
}

void plotMemoryOccupancyBenchmark(const std::string& inputFolder, const std::string& outputFolder)
{
  std::string graphPrefix = "plot-memory-occupancy-benchmark";
  std::vector<std::array<std::vector<int>, DataTypes>> dataReport = loadData(inputFolder + "MemoryOccupancy.txt");

  double binSize;
  std::array<double, BinNumber + 1> binsEdges;

  /// Tracklets Histograms
  binSize = 2e-03 / BinNumber;
  for (int iBin = 0; iBin <= BinNumber; ++iBin) {

    binsEdges[iBin] = iBin * binSize;
  }

  for(int iLayer = 0; iLayer < TrackletsNumber; ++iLayer) {

    std::string histogramId(graphPrefix + ".tracklets-graph-" + std::to_string(iLayer));
    std::string histogramTitle("Layer " + std::to_string(iLayer + 1) + " Tracklets Histogram");
    std::string outputFilename(outputFolder + "Layer" + std::to_string(iLayer + 1) + "TrackletsHistogram.pdf");

    TH1F trackletsHistogram(histogramId.c_str(), histogramTitle.c_str(), BinNumber, binsEdges.data());
    plotHistogram(trackletsHistogram, dataReport, TrackletsDataType, iLayer, outputFilename);
  }

  /// Cells Histograms
  binSize = 2e-08 / BinNumber;
  for (int iBin = 0; iBin <= BinNumber; ++iBin) {

    binsEdges[iBin] = iBin * binSize;
  }

  for(int iLayer = 0; iLayer < CellsNumber; ++iLayer) {

    std::string histogramId(graphPrefix + ".cells-graph-" + std::to_string(iLayer));
    std::string histogramTitle("Layer " + std::to_string(iLayer + 1) + " Cells Histogram");
    std::string outputFilename(outputFolder + "Layer" + std::to_string(iLayer + 1) + "CellsHistogram.pdf");

    TH1F cellsHistogram(histogramId.c_str(), histogramTitle.c_str(), BinNumber, binsEdges.data());
    plotHistogram(cellsHistogram, dataReport, CellsDataType, iLayer, outputFilename);
  }

  /// Roads Histogram
  binSize = 5e-14 / BinNumber;
  for (int iBin = 0; iBin <= BinNumber; ++iBin) {

    binsEdges[iBin] = iBin * binSize;
  }

  std::string histogramId(graphPrefix + ".roads-graph");
  std::string histogramTitle("Roads Histogram");
  std::string outputFilename(outputFolder + "RoadsHistogram.pdf");

  TH1F roadsHistogram(histogramId.c_str(), histogramTitle.c_str(), BinNumber, binsEdges.data());
  plotHistogram(roadsHistogram, dataReport, RoadsDataType, 0, outputFilename);
}

int main(int argc, char** argv)
{
  std::string inputFolder { "" };
  std::string outputFolder { "" };

  if (argv[1] != NULL) {

    inputFolder = std::string(argv[1]);
  }

  if (argv[2] != NULL) {

    outputFolder = std::string(argv[2]);
  }

  plotMemoryOccupancyBenchmark(inputFolder, outputFolder);
}

