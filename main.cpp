#include <iomanip>
#include <iostream>
#include <limits>
#include <fstream>
#include <vector>

#include "CAIOUtils.h"
#include "CATracker.h"

std::string getDirectory(const std::string& fname)
{
  size_t pos = fname.find_last_of("\\/");
  return (std::string::npos == pos) ? "" : fname.substr(0, pos + 1);
}

int main(int argc, char** argv)
{
  if (argv[1] == NULL) {

    std::cerr << "Please, provide a data file." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string eventsFileName(argv[1]);
  std::string benchmarkFolderName = getDirectory(eventsFileName);
  std::vector<CAEvent> events = CAIOUtils::loadEventData(eventsFileName);
  const int eventsNum = events.size();
  std::vector<std::unordered_map<int, CALabel>> labelsMap;
  bool createBenchmarkData = false;
  std::ofstream correctRoadsOutputStream;
  std::ofstream duplicateRoadsOutputStream;
  std::ofstream fakeRoadsOutputStream;

  int verticesNum = 0;
  for (int iEvent = 0; iEvent < eventsNum; ++iEvent) {

    verticesNum += events[iEvent].getPrimaryVerticesNum();
  }

  if (argv[2] != NULL) {

    std::string labelsFileName(argv[2]);

    createBenchmarkData = true;
    labelsMap = CAIOUtils::loadLabels(eventsNum, labelsFileName);

    correctRoadsOutputStream.open(benchmarkFolderName + "CorrectRoads.txt");
    duplicateRoadsOutputStream.open(benchmarkFolderName + "DuplicateRoads.txt");
    fakeRoadsOutputStream.open(benchmarkFolderName + "FakeRoads.txt");
  }

#if defined MEMORY_BENCHMARK
  std::ofstream memoryBenchmarkOutputStream;
  memoryBenchmarkOutputStream.open(benchmarkFolderName + "MemoryOccupancy.txt");
#endif


  for (int iEvent = 0; iEvent < eventsNum; ++iEvent) {

    CAEvent& currentEvent = events[iEvent];
    std::cout << "Processing event " << iEvent + 1 << std::endl;

#if defined DEBUG
    clock_t t1, t2;
    float totalTime = 0.f, minTime = std::numeric_limits<float>::max(), maxTime = -1;

    t1 = clock();


    std::vector<std::vector<CARoad>> roads = CATracker(currentEvent).clustersToTracksVerbose();

    t2 = clock();
    const float diff = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);

    totalTime += diff;

    if (minTime > diff)
      minTime = diff;
    if (maxTime < diff)
      maxTime = diff;

    std::cout << "Event " << iEvent + 1 << " processed in: " << diff << "ms" << std::endl << std::endl;
#elif defined MEMORY_BENCHMARK
    std::vector<std::vector<CARoad>> roads = CATracker(currentEvent).clustersToTracksMemoryBenchmark(memoryBenchmarkOutputStream);
#else
    std::vector<std::vector<CARoad>> roads = CATracker(currentEvent).clustersToTracks();
#endif

    if (createBenchmarkData) {

      for (auto& currentVertexRoads : roads) {

        CAIOUtils::writeRoadsReport(correctRoadsOutputStream, duplicateRoadsOutputStream, fakeRoadsOutputStream,
            currentVertexRoads, labelsMap[iEvent]);
      }
    }
  }

#if defined DEBUG
  std::cout << std::endl;
  std::cout << "Avg time: " << totalTime / verticesNum << "ms" << std::endl;
  std::cout << "Min time: " << minTime << "ms" << std::endl;
  std::cout << "Max time: " << maxTime << "ms" << std::endl;
#endif

  return 0;
}

