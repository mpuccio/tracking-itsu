#include "TCanvas.h"
#include "TGraph.h"
#include "TMath.h"
#include "TROOT.h"

void plotRoads(const char* inputFileName, const char* outputFileName, const char* histogramTitle) {

  const int binNumber = 100;
  const double ptcuth = 20;
  const double ptcutl = 0.01;

  double binsEdges[binNumber + 1];
  double a = TMath::Log(ptcuth/ptcutl) / binNumber;

  for (int iBin = 0; iBin <= binNumber; ++iBin) {

    binsEdges[iBin] = ptcutl * TMath::Exp(iBin * a);
  }

  TCanvas graphCanvas{};
  graphCanvas.SetGrid();
  graphCanvas.SetLogx();

  TH1F histogram("plot-transverse-momentum-benchmark.histogram", histogramTitle, binNumber, binsEdges);

  TGraphAsymmErrors graph(&histogram);
  graph.Draw("AL*");
  graphCanvas.Print(outputFileName);
}

void plotTransverseMomentumBenchmark() {

  plotRoads(
      "benchmarks/transverse_momentum/benchmark_data/CorrectRoads.txt",
      "benchmarks/transverse_momentum/CorrectRoadsBenchmark.pdf",
      "Correct Roads Histogram"
  );

  plotRoads(
      "benchmarks/transverse_momentum/benchmark_data/FakeRoads.txt",
      "benchmarks/transverse_momentum/FakeRoadsBenchmark.pdf",
      "Fake Roads Histogram"
  );
}

int main() {

  plotTransverseMomentumBenchmark();
}
