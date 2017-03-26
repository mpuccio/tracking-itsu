#include <iomanip>
#include <iostream>
#include <vector>

#include "CAEventLoader.h"
#include "CATracker.h"

int main(int argc, char** argv)
{
  clock_t t1, t2;

  for(int iIterations = 0; iIterations < 10; ++iIterations) {

    t1 = clock();

    if (argv[1] == NULL) {

      std::cerr << "Please, provide a data file." << std::endl;
      exit(EXIT_FAILURE);
    }

    std::string fileName(argv[1]);
    std::vector<CAEvent> events = CAEventLoader::loadEventData(fileName);

    int eventsNum = events.size();

    for(int iEvent = 0; iEvent < eventsNum; ++iEvent) {

      CATracker(events[iEvent]).clustersToTracks();
    }

    t2 = clock();

    float diff ((float)t2-(float)t1);
    std::cout << "Task completed in: " << diff/(CLOCKS_PER_SEC/1000) << "ms" << std::endl;
  }

  return 0;
}

