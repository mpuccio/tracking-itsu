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

void loadBinsFromFile(const char* fileName, TH1F& histogram)
{
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
}

void plotHistogramsRatio(TH1F& numeratorHistogram, TH1F& denominatorHistogram,
    std::array<float, BinNumber + 1>& binsEdges, const char* outputFileName, const char* histogramTitle)
{
  TCanvas graphCanvas { };
  graphCanvas.SetGrid();
  graphCanvas.SetLogx();

  TH1F histogramsRatio("plot-transverse-momentum-benchmark.histograms-ratio", histogramTitle, BinNumber,
      binsEdges.data());

  histogramsRatio.Divide(&numeratorHistogram, &denominatorHistogram);
  histogramsRatio.Draw();

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
  loadBinsFromFile("benchmarks/benchmark_data/labels.txt", generatedHistogram);

  TH1F correctHistogram("plot-transverse-momentum-benchmark.correct-histogram", "Correct Histogram", BinNumber,
      binsEdges.data());
  loadBinsFromFile("benchmarks/benchmark_data/CorrectRoads.txt", correctHistogram);

  TH1F duplicateHistogram("plot-transverse-momentum-benchmark.duplicate-histogram", "Duplicate Histogram", BinNumber,
      binsEdges.data());
  loadBinsFromFile("benchmarks/benchmark_data/DuplicateRoads.txt", duplicateHistogram);

  TH1F fakeHistogram("plot-transverse-momentum-benchmark.fake-histogram", "Fake Histogram", BinNumber,
      binsEdges.data());
  loadBinsFromFile("benchmarks/benchmark_data/FakeRoads.txt", fakeHistogram);

  plotHistogramsRatio(correctHistogram, generatedHistogram, binsEdges,
      "benchmarks/transverse_momentum/CorrectRoadsBenchmark.pdf", "Correct Roads Histogram");

  plotHistogramsRatio(duplicateHistogram, generatedHistogram, binsEdges,
      "benchmarks/transverse_momentum/DuplicateRoadsBenchmark.pdf", "Duplicate Roads Histogram");

  plotHistogramsRatio(fakeHistogram, generatedHistogram, binsEdges,
      "benchmarks/transverse_momentum/FakeRoadsBenchmark.pdf", "Fake Roads Histogram");
}

int main()
{

  plotTransverseMomentumBenchmark();
}
