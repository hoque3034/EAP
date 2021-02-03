#!/usr/bin/env bash
# run this program from the code folder
input_protein_list=../inputs/port_list.txt
echo $input_protein_list
line=5fcnB
#while IFS= read -r line
#do
#	python ./eap.py ${line} phi > ../outputs/${line}.phi
#	python ./eap.py ${line} psi > ../outputs/${line}.psi
#	python ./eap.py ${line} theta > ../outputs/${line}.theta
#	python ./eap.py ${line} tau > ../outputs/${line}.tau
#done < "$input_protein_list"
python3 ./sap.py ${line} phi > ../outputs/${line}.phi