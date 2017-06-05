#!/bin/bash

prob=(1 2)
prob2=(3 4)
ELEMPROB=${#prob[@]}
my_array=()

#Runs problems
for (( i=0;i<$ELEMPROB;i++)); do
    	var="$(python output.py ${prob[${i}]} ${prob2[${i}]})"
      my_array+=("$var")
done

len=${#my_array[@]}

Error=()
check=()


## Split printout and get E and check values
while IFS=' ' read -ra ADDR; do
     for i in "${ADDR[@]}"; do
         if [[ $i == *"E="* ]]; then
            echo "E is there!"
            echo $i
            Error+=($i)
          fi
          if [[ $i == *"check="* ]]; then
             echo "check is there!"
             echo $i
             check+=($i)
           fi
     done
done <<< "${my_array[1]}"

E2=()
Check=()

while IFS='=' read -ra ADDR; do
     for i in "${ADDR[@]}"; do
         E2+=($i)
     done
done <<< "${Error}"

while IFS='=' read -ra ADDR; do
     for i in "${ADDR[@]}"; do
         Check+=($i)
     done
done <<< "${check}"


errorlen=${#E2[@]}
for (( i=0;i<$errorlen;i++)); do
  echo ${E2[1][${i}]}
  echo ${Check[1][${i}]}
done


#string='My long string'
#if [[ $string == *"My long"* ]]; then
#  echo "It's there!"
#fi
