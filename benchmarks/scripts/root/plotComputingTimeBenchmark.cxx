#include <array>

#include "TMultiGraph.h"
#include "TGraph.h"

namespace {
constexpr int SamplesNumber { 4 };
constexpr std::array<float, SamplesNumber> VerticesNum { { 1, 2, 4, 5 } };
constexpr std::array<float, SamplesNumber> TrackletsMean { { 45.6006f, 152.505f, 556.192f, 851.774f } };
constexpr std::array<float, SamplesNumber> TrackletsMin { { 21.459f, 85.504f, 361.248f, 569.04f } };
constexpr std::array<float, SamplesNumber> TrackletsMax { { 63.232f, 195.068f, 665.488f, 1013.66f } };
constexpr std::array<float, SamplesNumber> CellsMean { { 28.7941f, 132.437f, 826.187f, 1493.8f } };
constexpr std::array<float, SamplesNumber> CellsMin { { 10.17f, 49.288f, 346.48f, 639.56f } };
constexpr std::array<float, SamplesNumber> CellsMax { { 44.8f, 187.506f, 1179.83f, 2096.4f } };
/*
constexpr int SamplesNumber { 4 };
constexpr std::array<float, SamplesNumber> VerticesNum { { 1, 2, 4, 5 } };
constexpr std::array<float, SamplesNumber> TrackletsMean { { 13.9352f, 42.7945f, 160.646f, 244.385f } };
constexpr std::array<float, SamplesNumber> TrackletsMin { { 5.984f, 24.156f, 105.838f, 170.472f } };
constexpr std::array<float, SamplesNumber> TrackletsMax { { 19.109f, 52.672f, 190.924f, 296.604f } };
constexpr std::array<float, SamplesNumber> CellsMean { { 8.07032f, 29.534f, 171.754f, 306.299f } };
constexpr std::array<float, SamplesNumber> CellsMin { { 2.801f, 11.49f, 70.778f, 129.568f } };
constexpr std::array<float, SamplesNumber> CellsMax { { 12.172f, 41.289f, 233.276f, 425.516f } };
*/
}

void plotMultiGraph(const std::string& outputFileName, const std::string& graphTitle, const std::array<float, SamplesNumber>& meanData,
	const std::array<float, SamplesNumber>& minData, const std::array<float, SamplesNumber>& maxData)
{
  TCanvas graphCanvas { };
  graphCanvas.SetGrid();

  TMultiGraph multiGraph{};
  multiGraph.SetTitle(graphTitle.c_str());

  TGraph meanGraph{SamplesNumber, VerticesNum.data(), meanData.data()};
  meanGraph.SetTitle("Mean Time");
  meanGraph.SetLineColor(kRed);
  meanGraph.SetMarkerStyle(20);
  meanGraph.SetMarkerColor(kRed);
  multiGraph.Add(&meanGraph);

  TGraph minGraph{SamplesNumber, VerticesNum.data(), minData.data()};
  minGraph.SetTitle("Min Time");
  minGraph.SetLineColor(kGreen);
  minGraph.SetMarkerStyle(20);
  minGraph.SetMarkerColor(kGreen);
  multiGraph.Add(&minGraph);

  TGraph maxGraph{SamplesNumber, VerticesNum.data(), maxData.data()};
  maxGraph.SetTitle("Max Time");
  maxGraph.SetLineColor(kBlue);
  maxGraph.SetMarkerStyle(20);
  maxGraph.SetMarkerColor(kBlue);
  multiGraph.Add(&maxGraph);

  multiGraph.Draw("ACP");
  multiGraph.GetXaxis()->SetTitle("#Vertices");
  multiGraph.GetYaxis()->SetTitle("Computing time (ms)");
  multiGraph.GetYaxis()->SetTitleOffset(1.5);

  graphCanvas.BuildLegend(0.1, 0.7, 0.3, 0.9);
  graphCanvas.Print(outputFileName.c_str());
}

void plotComputingTimeBenchmark(const std::string& outputFolder)
{
	plotMultiGraph(outputFolder + "TrackletsComputingTime.pdf", "Tracklets Computing Time", TrackletsMean, TrackletsMin, TrackletsMax);
	plotMultiGraph(outputFolder + "CellsComputingTime.pdf", "Cells Computing Time", CellsMean, CellsMin, CellsMax);
}
