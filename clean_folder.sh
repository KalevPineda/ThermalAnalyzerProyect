#!/bin/bash 

# limpíar archivos .h5 en la carpeta DataSets
echo "Limpiar archivos .h5 en DataSets"
find ./DataSets -type f -name "*.h5" -exec rm -f {} \;
# limpíar archivos .jpeg en la carpeta ThermalImages
echo "Limpiar archivos .jpeg en ThermalImages"
find ./ThermalImages -type f -name "*.jpeg" -exec rm -f {} \;

echo "\tLimpieza completada ..."

