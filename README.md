# Plasma_AI
This reprository contains the C addaptation of pytorch convolutive neuronal network trained to recognize written numbers.
# structure
- Python:
  This folder contains the python script that train the neuronal network and save the trained model (CNN.py) and a script that transforms the model into a C library.
- C:
  This folder contains the C sources files that allow to use the trained model. It is coded using fixed point approch.
- Backup:
  This folder contains the floating point version of the application.
-data:
  This folder contains the data base used to train the neuronal network.
# Tutorial
  To run the application run CNN.py then convertModel.py then create a build directory in C. In this directory do 'CMake ..' and 'make' then run matrices and you should see the results.
