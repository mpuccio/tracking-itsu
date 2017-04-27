#!/usr/bin/awk -f

#######################################################
#                                                     #
# This awk script can merge multiple events together, #
# simulating a pile-up condition. Variable pileUp     #
# represents the pile-up level: its value should be a #
# divisor of the total number of events.              #
#                                                     #
#######################################################

BEGIN {
  
  # pile-up level
  pileUp = 10;

  linesNum = 0;
  verticesNum = 0;
  primaryVertexIndex = -1;
  maxMonteCarloLabel = 0;
  monteCarloOffset = 0;
}

function roundUp(numToRound, multiple) {

  if(multiple == 0) return numToRound;
  remainder = numToRound % multiple;
  if(remainder == 0) return numToRound;
  return numToRound + multiple - remainder;
}

{
  if($1 == primaryVertexIndex) {
    
    maxMonteCarloLabel = roundUp(maxMonteCarloLabel, 100000);
    monteCarloOffset += maxMonteCarloLabel;
    maxMonteCarloLabel = 0;

    if(verticesNum == pileUp) {
      
      for(iLine = 0; iLine in lines; ++iLine) {
        
        print lines[iLine];
      }
      
      linesNum = 0;
      verticesNum = 0;
      monteCarloOffset = 0;
      delete lines;
    }

    print $0;
    ++verticesNum;
    
  } else {
    
    if($9 > maxMonteCarloLabel) {
      
      maxMonteCarloLabel = $9
    }

    $9 += monteCarloOffset

    lines[linesNum] = $0;
    ++linesNum;
  }
}

END {

  for(iLine = 0; iLine in lines; ++iLine) {

    print lines[iLine];
  }
}
