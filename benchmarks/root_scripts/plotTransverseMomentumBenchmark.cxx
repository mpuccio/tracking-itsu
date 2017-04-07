#include <array>
#include <cmath>
#include <fstream>
#include <string>

#include "TCanvas.h"
#include "TGraph.h"
#include "TROOT.h"

namespace {
constexpr int BinNumber { 100 };
constexpr float MaxTransverseMomentum { 20 };
constexpr float MinTransverseMomentum { 0.01 };
constexpr int EventLabelsSeparator { -1 };
constexpr int PionCode { 211 };
}

std::array<int, BinNumber> loadBinsFromFile(const char* fileName, TH1F& histogram)
{

  std::array<int, BinNumber> tracksPerBin;
  tracksPerBin.fill(0);

  std::ifstream inputStream;
  std::string line;
  inputStream.open(fileName);
  int monteCarloId, pdgCode, numberOfClusters;
  float transverseMomentum, phiCoordinate, pseudorapidity;

  while (std::getline(inputStream, line)) {

    std::istringstream inputStringStream(line);

    if (inputStringStream >> monteCarloId) {

      if (monteCarloId == EventLabelsSeparator) {

        continue;

      } else {

        if (inputStringStream >> transverseMomentum >> phiCoordinate >> pseudorapidity >> pdgCode >> numberOfClusters) {

          if (std::abs(pdgCode) == PionCode && numberOfClusters == 7) {

            histogram.Fill(transverseMomentum);
          }
        }
      }
    }
  }

  return tracksPerBin;
}

void plotRoads(TH1F& generatedHistogram, std::array<float, BinNumber + 1>& binsEdges, const char* inputFileName,
    const char* outputFileName, const char* histogramTitle)
{

  TCanvas graphCanvas { };
  graphCanvas.SetGrid();
  graphCanvas.SetLogx();

  TH1F foundHistogram("plot-transverse-momentum-benchmark.found-histogram", histogramTitle, BinNumber,
      binsEdges.data());
  std::array<int, BinNumber> foundTracksPerBin = loadBinsFromFile(inputFileName, foundHistogram);
  foundHistogram.Divide(&generatedHistogram);
  foundHistogram.Draw();

  graphCanvas.Print(outputFileName);
}

void plotTransverseMomentumBenchmark()
{

  float binSize = std::log(MaxTransverseMomentum / MinTransverseMomentum) / BinNumber;
  std::array<float, BinNumber + 1> binsEdges;

  for (int iBin = 0; iBin <= BinNumber; ++iBin) {

    binsEdges[iBin] = MinTransverseMomentum * std::exp(iBin * binSize);
  }

  TH1F generatedHistogram("plot-transverse-momentum-benchmark.generated-histogram", "Generated Histogram", BinNumber,
      binsEdges.data());

  std::array<int, BinNumber> generatedTracksPerBin = loadBinsFromFile("benchmarks/benchmark_data/labels.txt",
      generatedHistogram);

  plotRoads(generatedHistogram, binsEdges, "benchmarks/benchmark_data/CorrectRoads.txt",
      "benchmarks/transverse_momentum/CorrectRoadsBenchmark.pdf", "Correct Roads Histogram");

  plotRoads(generatedHistogram, binsEdges, "benchmarks/benchmark_data/DuplicateRoads.txt",
      "benchmarks/transverse_momentum/DuplicateRoadsBenchmark.pdf", "Duplicate Roads Histogram");

  plotRoads(generatedHistogram, binsEdges, "benchmarks/benchmark_data/FakeRoads.txt",
      "benchmarks/transverse_momentum/FakeRoadsBenchmark.pdf", "Fake Roads Histogram");
}

int main()
{

  plotTransverseMomentumBenchmark();
}
