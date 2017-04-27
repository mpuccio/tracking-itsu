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

void loadBinsFromFile(const std::string& fileName, TH1F& histogram)
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
    std::array<float, BinNumber + 1>& binsEdges, const std::string& outputFileName, const char* histogramTitle)
{
  TCanvas graphCanvas { };
  graphCanvas.SetGrid();
  graphCanvas.SetLogx();

  TH1F histogramsRatio("plot-transverse-momentum-benchmark.histograms-ratio", histogramTitle, BinNumber,
      binsEdges.data());

  histogramsRatio.Divide(&numeratorHistogram, &denominatorHistogram);
  histogramsRatio.Draw();

  graphCanvas.Print(outputFileName.c_str());
}

void plotTransverseMomentumBenchmark(const std::string& inputFolder, const std::string& outputFolder)
{
  float binSize = std::log(MaxTransverseMomentum / MinTransverseMomentum) / BinNumber;
  std::array<float, BinNumber + 1> binsEdges;

  for (int iBin = 0; iBin <= BinNumber; ++iBin) {

    binsEdges[iBin] = MinTransverseMomentum * std::exp(iBin * binSize);
  }

  TH1F generatedHistogram("plot-transverse-momentum-benchmark.generated-histogram", "Generated Histogram", BinNumber,
      binsEdges.data());
  loadBinsFromFile(inputFolder + "merged_labels.txt", generatedHistogram);

  TH1F correctHistogram("plot-transverse-momentum-benchmark.correct-histogram", "Correct Histogram", BinNumber,
      binsEdges.data());
  loadBinsFromFile(inputFolder + "CorrectRoads.txt", correctHistogram);

  plotHistogramsRatio(correctHistogram, generatedHistogram, binsEdges, outputFolder + "CorrectRoadsBenchmark.pdf",
      "Correct Roads Histogram");

  TH1F duplicateHistogram("plot-transverse-momentum-benchmark.duplicate-histogram", "Duplicate Histogram", BinNumber,
      binsEdges.data());
  loadBinsFromFile(inputFolder + "DuplicateRoads.txt", duplicateHistogram);

  plotHistogramsRatio(duplicateHistogram, generatedHistogram, binsEdges, outputFolder + "DuplicateRoadsBenchmark.pdf",
      "Duplicate Roads Histogram");

  TH1F fakeHistogram("plot-transverse-momentum-benchmark.fake-histogram", "Fake Histogram", BinNumber,
      binsEdges.data());
  loadBinsFromFile(inputFolder + "FakeRoads.txt", fakeHistogram);

  plotHistogramsRatio(fakeHistogram, generatedHistogram, binsEdges, outputFolder + "FakeRoadsBenchmark.pdf",
      "Fake Roads Histogram");

  TH1F duplicateAndCorrectHistogram("plot-transverse-momentum-benchmark.duplicate-and-correct-histogram",
      "Duplicate And Correct Histogram", BinNumber, binsEdges.data());
  duplicateAndCorrectHistogram.Add(&correctHistogram, &duplicateHistogram);

  plotHistogramsRatio(duplicateHistogram, duplicateAndCorrectHistogram, binsEdges,
      outputFolder + "DuplicateOverCorrectRoadsBenchmark.pdf", "Duplicate Over Correct Roads Histogram");
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

  plotTransverseMomentumBenchmark(inputFolder, outputFolder);
}
