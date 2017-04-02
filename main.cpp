#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "CAEventLoader.h"
#include "CATracker.h"

int main(int argc, char** argv)
{
  if (argv[1] == NULL) {

    std::cerr << "Please, provide a data file." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string fileName(argv[1]);
  std::vector<CAEvent> events = CAEventLoader::loadEventData(fileName);
  const int eventsNum = events.size();

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

    CATracker(currentEvent).clustersToTracksVerbose();

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

