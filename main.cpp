#include <iomanip>
#include <iostream>
#include <limits>
#include <fstream>
#include <vector>

#include "CAIOUtils.h"
#include "CATracker.h"

int main(int argc, char** argv)
{
  if (argv[1] == NULL) {

    std::cerr << "Please, provide a data file." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string eventsFileName(argv[1]);
  std::vector<CAEvent> events = CAIOUtils::loadEventData(eventsFileName);
  const int eventsNum = events.size();
  std::vector<std::unordered_map<int, CALabel>> labelsMap;

  bool createBenchmarkData = false;
  std::ofstream correctRoadsOutputStream;
  std::ofstream duplicateRoadsOutputStream;
  std::ofstream fakeRoadsOutputStream;

  if (argv[2] != NULL) {

    createBenchmarkData = true;

    std::string labelsFileName(argv[2]);
    labelsMap = CAIOUtils::loadLabels(eventsNum, labelsFileName);

    correctRoadsOutputStream.open("benchmarks/benchmark_data/CorrectRoads.txt");
    duplicateRoadsOutputStream.open("benchmarks/benchmark_data/DuplicateRoads.txt");
    fakeRoadsOutputStream.open("benchmarks/benchmark_data/FakeRoads.txt");
  }

  clock_t t1, t2;
  float totalTime = 0.f, minTime = std::numeric_limits<float>::max(), maxTime = -1;

  for (int iEvent = 0; iEvent < eventsNum; ++iEvent) {

    CAEvent& currentEvent = events[iEvent];

    std::cout << "Sorting clusters for event " << iEvent + 1 << std::endl;
    t1 = clock();

    currentEvent.sortClusters();

    t2 = clock();
    const float sortingDiff = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
    std::cout << "Clusters sorted in " << sortingDiff << "ms" << std::endl;

    std::cout << "Processing event " << iEvent + 1 << ":" << std::endl;
    t1 = clock();

    std::vector<CARoad> roads = CATracker(currentEvent).clustersToTracksVerbose();

    if (createBenchmarkData) {

      CAIOUtils::writeRoadsReport(correctRoadsOutputStream, duplicateRoadsOutputStream, fakeRoadsOutputStream, roads,
          labelsMap[iEvent]);
    }

    t2 = clock();
    const float diff = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);

    totalTime += diff;

    if (minTime > diff)
      minTime = diff;
    if (maxTime < diff)
      maxTime = diff;

    std::cout << "Event " << iEvent + 1 << " processed in: " << diff << "ms" << std::endl << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Avg time: " << totalTime / eventsNum << "ms" << std::endl;
  std::cout << "Min time: " << minTime << "ms" << std::endl;
  std::cout << "Max time: " << maxTime << "ms" << std::endl;

  return 0;
}

