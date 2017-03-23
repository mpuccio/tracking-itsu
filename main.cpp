#include <iostream>
#include <vector>

#include "CAEventLoader.h"

int main(int argc, char** argv)
{

  if (argv[1] == NULL) {

    std::cerr << "Please, provide a data file." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string fileName(argv[1]);
  std::vector<CAEvent> events = CAEventLoader::loadEventData(fileName);

  std::cout << "Hello World!!" << std::endl;
}

